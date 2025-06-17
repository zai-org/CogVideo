import logging
import math
import re
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.distributed
from packaging import version

from vae_modules.ema import LitEma
from sgm.util import (
    instantiate_from_config,
    get_obj_from_str,
    default,
    is_context_parallel_initialized,
    initialize_context_parallel,
    get_context_parallel_group,
    get_context_parallel_group_rank,
)
from vae_modules.cp_enc_dec import _conv_split, _conv_gather

logpy = logging.getLogger(__name__)


class AbstractAutoencoder(pl.LightningModule):
    """
    This is the base class for all autoencoders, including image autoencoders, image autoencoders with discriminators,
    unCLIP models, etc. Hence, it is fairly general, and specific features
    (e.g. discriminator training, encoding, decoding) must be implemented in subclasses.
    """

    def __init__(
        self,
        ema_decay: Union[None, float] = None,
        monitor: Union[None, str] = None,
        input_key: str = "jpg",
    ):
        super().__init__()

        self.input_key = input_key
        self.use_ema = ema_decay is not None
        if monitor is not None:
            self.monitor = monitor

        if self.use_ema:
            self.model_ema = LitEma(self, decay=ema_decay)
            logpy.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            self.automatic_optimization = False

    def apply_ckpt(self, ckpt: Union[None, str, dict]):
        if ckpt is None:
            return
        if isinstance(ckpt, str):
            ckpt = {
                "target": "sgm.modules.checkpoint.CheckpointEngine",
                "params": {"ckpt_path": ckpt},
            }
        engine = instantiate_from_config(ckpt)
        engine(self)

    @abstractmethod
    def get_input(self, batch) -> Any:
        raise NotImplementedError()

    def on_train_batch_end(self, *args, **kwargs):
        # for EMA computation
        if self.use_ema:
            self.model_ema(self)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                logpy.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    logpy.info(f"{context}: Restored training weights")

    @abstractmethod
    def encode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("encode()-method of abstract base class called")

    @abstractmethod
    def decode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("decode()-method of abstract base class called")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        logpy.info(f"loading >>> {cfg['target']} <<< optimizer from config")
        return get_obj_from_str(cfg["target"])(params, lr=lr, **cfg.get("params", dict()))

    def configure_optimizers(self) -> Any:
        raise NotImplementedError()


class AutoencodingEngine(AbstractAutoencoder):
    """
    Base class for all image autoencoders that we train, like VQGAN or AutoencoderKL
    (we also restore them explicitly as special cases for legacy reasons).
    Regularizations such as KL or VQ are moved to the regularizer class.
    """

    def __init__(
        self,
        *args,
        encoder_config: Dict,
        decoder_config: Dict,
        loss_config: Dict,
        regularizer_config: Dict,
        optimizer_config: Union[Dict, None] = None,
        lr_g_factor: float = 1.0,
        trainable_ae_params: Optional[List[List[str]]] = None,
        ae_optimizer_args: Optional[List[dict]] = None,
        trainable_disc_params: Optional[List[List[str]]] = None,
        disc_optimizer_args: Optional[List[dict]] = None,
        disc_start_iter: int = 0,
        diff_boost_factor: float = 3.0,
        ckpt_engine: Union[None, str, dict] = None,
        ckpt_path: Optional[str] = None,
        additional_decode_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False  # pytorch lightning

        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.loss = instantiate_from_config(loss_config)
        self.regularization = instantiate_from_config(regularizer_config)
        self.optimizer_config = default(optimizer_config, {"target": "torch.optim.Adam"})
        self.diff_boost_factor = diff_boost_factor
        self.disc_start_iter = disc_start_iter
        self.lr_g_factor = lr_g_factor
        self.trainable_ae_params = trainable_ae_params
        if self.trainable_ae_params is not None:
            self.ae_optimizer_args = default(
                ae_optimizer_args,
                [{} for _ in range(len(self.trainable_ae_params))],
            )
            assert len(self.ae_optimizer_args) == len(self.trainable_ae_params)
        else:
            self.ae_optimizer_args = [{}]  # makes type consistent

        self.trainable_disc_params = trainable_disc_params
        if self.trainable_disc_params is not None:
            self.disc_optimizer_args = default(
                disc_optimizer_args,
                [{} for _ in range(len(self.trainable_disc_params))],
            )
            assert len(self.disc_optimizer_args) == len(self.trainable_disc_params)
        else:
            self.disc_optimizer_args = [{}]  # makes type consistent

        if ckpt_path is not None:
            assert ckpt_engine is None, "Can't set ckpt_engine and ckpt_path"
            logpy.warning("Checkpoint path is deprecated, use `checkpoint_egnine` instead")
        self.apply_ckpt(default(ckpt_path, ckpt_engine))
        self.additional_decode_keys = set(default(additional_decode_keys, []))

    def get_input(self, batch: Dict) -> torch.Tensor:
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in channels-first
        # format (e.g., bchw instead if bhwc)
        return batch[self.input_key]

    def get_autoencoder_params(self) -> list:
        params = []
        if hasattr(self.loss, "get_trainable_autoencoder_parameters"):
            params += list(self.loss.get_trainable_autoencoder_parameters())
        if hasattr(self.regularization, "get_trainable_parameters"):
            params += list(self.regularization.get_trainable_parameters())
        params = params + list(self.encoder.parameters())
        params = params + list(self.decoder.parameters())
        return params

    def get_discriminator_params(self) -> list:
        if hasattr(self.loss, "get_trainable_parameters"):
            params = list(self.loss.get_trainable_parameters())  # e.g., discriminator
        else:
            params = []
        return params

    def get_last_layer(self):
        return self.decoder.get_last_layer()

    def encode(
        self,
        x: torch.Tensor,
        return_reg_log: bool = False,
        unregularized: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        z = self.encoder(x, **kwargs)
        if unregularized:
            return z, dict()
        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.decoder(z, **kwargs)
        return x

    def forward(
        self, x: torch.Tensor, **additional_decode_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        z, reg_log = self.encode(x, return_reg_log=True)
        dec = self.decode(z, **additional_decode_kwargs)
        return z, dec, reg_log

    def inner_training_step(
        self, batch: dict, batch_idx: int, optimizer_idx: int = 0
    ) -> torch.Tensor:
        x = self.get_input(batch)
        additional_decode_kwargs = {
            key: batch[key] for key in self.additional_decode_keys.intersection(batch)
        }
        z, xrec, regularization_log = self(x, **additional_decode_kwargs)
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
                "optimizer_idx": optimizer_idx,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "train",
                "regularization_log": regularization_log,
                "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()

        if optimizer_idx == 0:
            # autoencode
            out_loss = self.loss(x, xrec, **extra_info)
            if isinstance(out_loss, tuple):
                aeloss, log_dict_ae = out_loss
            else:
                # simple loss function
                aeloss = out_loss
                log_dict_ae = {"train/loss/rec": aeloss.detach()}

            self.log_dict(
                log_dict_ae,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
            )
            self.log(
                "loss",
                aeloss.mean().detach(),
                prog_bar=True,
                logger=False,
                on_epoch=False,
                on_step=True,
            )
            return aeloss
        elif optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(x, xrec, **extra_info)
            # -> discriminator always needs to return a tuple
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss
        else:
            raise NotImplementedError(f"Unknown optimizer {optimizer_idx}")

    def training_step(self, batch: dict, batch_idx: int):
        opts = self.optimizers()
        if not isinstance(opts, list):
            # Non-adversarial case
            opts = [opts]
        optimizer_idx = batch_idx % len(opts)
        if self.global_step < self.disc_start_iter:
            optimizer_idx = 0
        opt = opts[optimizer_idx]
        opt.zero_grad()
        with opt.toggle_model():
            loss = self.inner_training_step(batch, batch_idx, optimizer_idx=optimizer_idx)
            self.manual_backward(loss)
        opt.step()

    def validation_step(self, batch: dict, batch_idx: int) -> Dict:
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
            log_dict.update(log_dict_ema)
        return log_dict

    def _validation_step(self, batch: dict, batch_idx: int, postfix: str = "") -> Dict:
        x = self.get_input(batch)

        z, xrec, regularization_log = self(x)
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
                "optimizer_idx": 0,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "val" + postfix,
                "regularization_log": regularization_log,
                "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()
        out_loss = self.loss(x, xrec, **extra_info)
        if isinstance(out_loss, tuple):
            aeloss, log_dict_ae = out_loss
        else:
            # simple loss function
            aeloss = out_loss
            log_dict_ae = {f"val{postfix}/loss/rec": aeloss.detach()}
        full_log_dict = log_dict_ae

        if "optimizer_idx" in extra_info:
            extra_info["optimizer_idx"] = 1
            discloss, log_dict_disc = self.loss(x, xrec, **extra_info)
            full_log_dict.update(log_dict_disc)
        self.log(
            f"val{postfix}/loss/rec",
            log_dict_ae[f"val{postfix}/loss/rec"],
            sync_dist=True,
        )
        self.log_dict(full_log_dict, sync_dist=True)
        return full_log_dict

    def get_param_groups(
        self, parameter_names: List[List[str]], optimizer_args: List[dict]
    ) -> Tuple[List[Dict[str, Any]], int]:
        groups = []
        num_params = 0
        for names, args in zip(parameter_names, optimizer_args):
            params = []
            for pattern_ in names:
                pattern_params = []
                pattern = re.compile(pattern_)
                for p_name, param in self.named_parameters():
                    if re.match(pattern, p_name):
                        pattern_params.append(param)
                        num_params += param.numel()
                if len(pattern_params) == 0:
                    logpy.warning(f"Did not find parameters for pattern {pattern_}")
                params.extend(pattern_params)
            groups.append({"params": params, **args})
        return groups, num_params

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        if self.trainable_ae_params is None:
            ae_params = self.get_autoencoder_params()
        else:
            ae_params, num_ae_params = self.get_param_groups(
                self.trainable_ae_params, self.ae_optimizer_args
            )
            logpy.info(f"Number of trainable autoencoder parameters: {num_ae_params:,}")
        if self.trainable_disc_params is None:
            disc_params = self.get_discriminator_params()
        else:
            disc_params, num_disc_params = self.get_param_groups(
                self.trainable_disc_params, self.disc_optimizer_args
            )
            logpy.info(f"Number of trainable discriminator parameters: {num_disc_params:,}")
        opt_ae = self.instantiate_optimizer_from_config(
            ae_params,
            default(self.lr_g_factor, 1.0) * self.learning_rate,
            self.optimizer_config,
        )
        opts = [opt_ae]
        if len(disc_params) > 0:
            opt_disc = self.instantiate_optimizer_from_config(
                disc_params, self.learning_rate, self.optimizer_config
            )
            opts.append(opt_disc)

        return opts

    @torch.no_grad()
    def log_images(
        self, batch: dict, additional_log_kwargs: Optional[Dict] = None, **kwargs
    ) -> dict:
        log = dict()
        additional_decode_kwargs = {}
        x = self.get_input(batch)
        additional_decode_kwargs.update(
            {key: batch[key] for key in self.additional_decode_keys.intersection(batch)}
        )

        _, xrec, _ = self(x, **additional_decode_kwargs)
        log["inputs"] = x
        log["reconstructions"] = xrec
        diff = 0.5 * torch.abs(torch.clamp(xrec, -1.0, 1.0) - x)
        diff.clamp_(0, 1.0)
        log["diff"] = 2.0 * diff - 1.0
        # diff_boost shows location of small errors, by boosting their
        # brightness.
        log["diff_boost"] = 2.0 * torch.clamp(self.diff_boost_factor * diff, 0.0, 1.0) - 1
        if hasattr(self.loss, "log_images"):
            log.update(self.loss.log_images(x, xrec))
        with self.ema_scope():
            _, xrec_ema, _ = self(x, **additional_decode_kwargs)
            log["reconstructions_ema"] = xrec_ema
            diff_ema = 0.5 * torch.abs(torch.clamp(xrec_ema, -1.0, 1.0) - x)
            diff_ema.clamp_(0, 1.0)
            log["diff_ema"] = 2.0 * diff_ema - 1.0
            log["diff_boost_ema"] = (
                2.0 * torch.clamp(self.diff_boost_factor * diff_ema, 0.0, 1.0) - 1
            )
        if additional_log_kwargs:
            additional_decode_kwargs.update(additional_log_kwargs)
            _, xrec_add, _ = self(x, **additional_decode_kwargs)
            log_str = "reconstructions-" + "-".join(
                [f"{key}={additional_log_kwargs[key]}" for key in additional_log_kwargs]
            )
            log[log_str] = xrec_add
        return log


class AutoencodingEngineLegacy(AutoencodingEngine):
    def __init__(self, embed_dim: int, **kwargs):
        self.max_batch_size = kwargs.pop("max_batch_size", None)
        ddconfig = kwargs.pop("ddconfig")
        ckpt_path = kwargs.pop("ckpt_path", None)
        ckpt_engine = kwargs.pop("ckpt_engine", None)
        super().__init__(
            encoder_config={
                "target": "sgm.modules.diffusionmodules.model.Encoder",
                "params": ddconfig,
            },
            decoder_config={
                "target": "sgm.modules.diffusionmodules.model.Decoder",
                "params": ddconfig,
            },
            **kwargs,
        )
        self.quant_conv = torch.nn.Conv2d(
            (1 + ddconfig["double_z"]) * ddconfig["z_channels"],
            (1 + ddconfig["double_z"]) * embed_dim,
            1,
        )
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        self.apply_ckpt(default(ckpt_path, ckpt_engine))

    def get_autoencoder_params(self) -> list:
        params = super().get_autoencoder_params()
        return params

    def encode(
        self, x: torch.Tensor, return_reg_log: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        if self.max_batch_size is None:
            z = self.encoder(x)
            z = self.quant_conv(z)
        else:
            N = x.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            z = list()
            for i_batch in range(n_batches):
                z_batch = self.encoder(x[i_batch * bs : (i_batch + 1) * bs])
                z_batch = self.quant_conv(z_batch)
                z.append(z_batch)
            z = torch.cat(z, 0)

        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: torch.Tensor, **decoder_kwargs) -> torch.Tensor:
        if self.max_batch_size is None:
            dec = self.post_quant_conv(z)
            dec = self.decoder(dec, **decoder_kwargs)
        else:
            N = z.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            dec = list()
            for i_batch in range(n_batches):
                dec_batch = self.post_quant_conv(z[i_batch * bs : (i_batch + 1) * bs])
                dec_batch = self.decoder(dec_batch, **decoder_kwargs)
                dec.append(dec_batch)
            dec = torch.cat(dec, 0)

        return dec


class AutoencoderKL(AutoencodingEngineLegacy):
    def __init__(self, **kwargs):
        if "lossconfig" in kwargs:
            kwargs["loss_config"] = kwargs.pop("lossconfig")
        super().__init__(
            regularizer_config={
                "target": ("sgm.modules.autoencoding.regularizers" ".DiagonalGaussianRegularizer")
            },
            **kwargs,
        )


class IdentityFirstStage(AbstractAutoencoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_input(self, x: Any) -> Any:
        return x

    def encode(self, x: Any, *args, **kwargs) -> Any:
        return x

    def decode(self, x: Any, *args, **kwargs) -> Any:
        return x


class VideoAutoencodingEngine(AutoencodingEngine):
    def __init__(
        self,
        ckpt_path: Union[None, str] = None,
        ignore_keys: Union[Tuple, list] = (),
        image_video_weights=[1, 1],
        only_train_decoder=False,
        context_parallel_size=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.context_parallel_size = context_parallel_size
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def log_videos(
        self, batch: dict, additional_log_kwargs: Optional[Dict] = None, **kwargs
    ) -> dict:
        return self.log_images(batch, additional_log_kwargs, **kwargs)

    def get_input(self, batch: dict) -> torch.Tensor:
        if self.context_parallel_size > 0:
            if not is_context_parallel_initialized():
                initialize_context_parallel(self.context_parallel_size)

            batch = batch[self.input_key]

            global_src_rank = get_context_parallel_group_rank() * self.context_parallel_size
            torch.distributed.broadcast(
                batch, src=global_src_rank, group=get_context_parallel_group()
            )

            batch = _conv_split(batch, dim=2, kernel_size=1)
            return batch

        return batch[self.input_key]

    def apply_ckpt(self, ckpt: Union[None, str, dict]):
        if ckpt is None:
            return
        self.init_from_ckpt(ckpt)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print("Missing keys: ", missing_keys)
        print("Unexpected keys: ", unexpected_keys)
        print(f"Restored from {path}")


class VideoAutoencoderInferenceWrapper(VideoAutoencodingEngine):
    def __init__(
        self,
        cp_size=0,
        *args,
        **kwargs,
    ):
        self.cp_size = cp_size
        return super().__init__(*args, **kwargs)

    def encode(
        self,
        x: torch.Tensor,
        return_reg_log: bool = False,
        unregularized: bool = False,
        input_cp: bool = False,
        output_cp: bool = False,
        use_cp: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        if self.cp_size <= 1:
            use_cp = False
        if self.cp_size > 0 and use_cp and not input_cp:
            if not is_context_parallel_initialized:
                initialize_context_parallel(self.cp_size)

            global_src_rank = get_context_parallel_group_rank() * self.cp_size
            torch.distributed.broadcast(x, src=global_src_rank, group=get_context_parallel_group())

            x = _conv_split(x, dim=2, kernel_size=1)

        if return_reg_log:
            z, reg_log = super().encode(x, return_reg_log, unregularized, use_cp=use_cp)
        else:
            z = super().encode(x, return_reg_log, unregularized, use_cp=use_cp)

        if self.cp_size > 0 and use_cp and not output_cp:
            z = _conv_gather(z, dim=2, kernel_size=1)

        if return_reg_log:
            return z, reg_log
        return z

    def decode(
        self,
        z: torch.Tensor,
        input_cp: bool = False,
        output_cp: bool = False,
        use_cp: bool = True,
        **kwargs,
    ):
        if self.cp_size <= 1:
            use_cp = False
        if self.cp_size > 0 and use_cp and not input_cp:
            if not is_context_parallel_initialized:
                initialize_context_parallel(self.cp_size)

            global_src_rank = get_context_parallel_group_rank() * self.cp_size
            torch.distributed.broadcast(z, src=global_src_rank, group=get_context_parallel_group())

            z = _conv_split(z, dim=2, kernel_size=1)

        x = super().decode(z, use_cp=use_cp, **kwargs)

        if self.cp_size > 0 and use_cp and not output_cp:
            x = _conv_gather(x, dim=2, kernel_size=1)
        return x

    def forward(
        self,
        x: torch.Tensor,
        input_cp: bool = False,
        latent_cp: bool = False,
        output_cp: bool = False,
        **additional_decode_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        z, reg_log = self.encode(x, return_reg_log=True, input_cp=input_cp, output_cp=latent_cp)
        dec = self.decode(z, input_cp=latent_cp, output_cp=output_cp, **additional_decode_kwargs)
        return z, dec, reg_log
