from .spmuple import SPMuple
from .spmuple2 import SPMuple2


class SPMupleOnset(SPMuple2):
    def _tweak_config_before_creating_voc(self):
        super()._tweak_config_before_creating_voc()

        self.config.additional_params["use_position_shifts"] = True
        self.config.additional_params["use_onset_indices"] = True

        self.config.additional_params["onset_tempos"] = True


class SPMupleBeat(SPMuple):
    def _tweak_config_before_creating_voc(self):
        super()._tweak_config_before_creating_voc()

        self.config.additional_params["use_position_shifts"] = True
        self.config.additional_params["use_onset_indices"] = True
        self.config.additional_params["rel_onset_dev"] = True
        self.config.additional_params["rel_perf_duration"] = True

        self.config.additional_params["bar_tempos"] = False


class SPMupleBar(SPMuple):
    def _tweak_config_before_creating_voc(self):
        super()._tweak_config_before_creating_voc()

        self.config.additional_params["use_position_shifts"] = True
        self.config.additional_params["use_onset_indices"] = True
        self.config.additional_params["rel_onset_dev"] = True
        self.config.additional_params["rel_perf_duration"] = True

        self.config.additional_params["bar_tempos"] = True


class SPMupleWindow(SPMuple2):
    def _tweak_config_before_creating_voc(self):
        super()._tweak_config_before_creating_voc()

        self.config.additional_params["use_position_shifts"] = True
        self.config.additional_params["use_onset_indices"] = True

        self.config.additional_params["use_quantized_tempos"] = True
        self.config.additional_params["decode_recompute_tempos"] = False


class SPMupleWindowRecompute(SPMuple2):
    def _tweak_config_before_creating_voc(self):
        super()._tweak_config_before_creating_voc()

        self.config.additional_params["use_position_shifts"] = True
        self.config.additional_params["use_onset_indices"] = True

        self.config.additional_params["use_quantized_tempos"] = self.config.additional_params.get(
            "use_quantized_tempos", True
        )
        self.config.additional_params["decode_recompute_tempos"] = True
