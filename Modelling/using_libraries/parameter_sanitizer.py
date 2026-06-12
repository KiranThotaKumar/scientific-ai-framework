#using_libraries\parameter_sanitizer.py

class ParameterSanitizer:

    def sanitize(self, intent):

        params = intent.parameters

        emin = params.get("emin")
        emax = params.get("emax")

        if emin is not None and emax is not None:

            if emin > emax:

                params["emin"], params["emax"] = (
                    params["emax"],
                    params["emin"]
                )

        if (
            intent.domain == "hydrogen"
            and intent.action == "forward"
        ):

            self._sanitize_hydrogen_forward(
                params
            )

        if (
            intent.domain == "single_qubit"
            and intent.action == "infer_parameters"
            ):

                self._sanitize_single_qubit_inference(
                    params
                )

        return intent


    def _sanitize_single_qubit_inference(self, params):


        if "ntimes" not in params:
            params["ntimes"] = 200
            
        
        if "noise_std" not in params:
            params["noise_std"] = 0.02

    def _sanitize_hydrogen_forward(self, params):

        if params.get("spectrum_mode") == "none":
            params["spectrum_mode"] = "emission"

        if "nbins" not in params:
            params["nbins"] = 200

        series = params.get("series")

        if series and "transitions" not in params:

            params["transitions"] = \
                self._build_transitions(series)

    def _build_transitions(self, series):

        series = series.lower()

        if series == "lyman":

            n_l = 1
            n_upper = range(2, 5)

        elif series == "balmer":

            n_l = 2
            n_upper = range(3, 6)

        elif series == "paschen":

            n_l = 3
            n_upper = range(4, 7)

        elif series == "brackett":

            n_l = 4
            n_upper = range(5, 8)

        elif series == "pfund":

            n_l = 5
            n_upper = range(6, 9)

        else:
            return []

        transitions = []

        for n_u in n_upper:

            transitions.append(
                (n_u, n_l)
            )

        return transitions