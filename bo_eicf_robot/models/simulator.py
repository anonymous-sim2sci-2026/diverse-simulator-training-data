"""
NMMA simulation runner
"""

import os
import glob
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from typing import Tuple


class SimulationRunner:
    """
    Wraps the NMMA lightcurve-analysis CLI to run real kilonova simulations.
    """

    def __init__(self, config, detection_calc=None):
        self.config = config
        self.exposure_times = config.EXPOSURE_TIMES
        self.filters = config.FILTERS
        self.svd_path = config.SVD_PATH
        self.prior_file = config.PRIOR_FILE
        self.error_threshold = config.ERROR_THRESHOLD
        self.detection_calc = detection_calc

    def calculate_chirp_mass(self, mass1, mass2):
        """Calculate chirp mass from component masses"""
        return (mass1 * mass2) ** 0.6 / (mass1 + mass2) ** 0.2

    def map_gw_to_kilonova(self, gw_params):
        """
        Map GW parameters to kilonova model parameters.
        """
        # Calculate chirp mass if not provided
        if 'chirp_mass' not in gw_params and 'mass1' in gw_params and 'mass2' in gw_params:
            gw_params['chirp_mass'] = self.calculate_chirp_mass(
                gw_params['mass1'], gw_params['mass2']
            )

        # Get parameters with defaults
        P_NS = gw_params.get('HasNS', 0.5)
        P_EMB = gw_params.get('HasRemnant', 0.5)
        P_MassGap = gw_params.get('HasMassGap', 0.5)

        # Build kilonova parameters
        kn_params = {
            'mass_1': gw_params.get('mass1', 1.4),
            'mass_2': gw_params.get('mass2', 1.4),
            'lambda_1': 100 + 400 * P_NS,
            'lambda_2': 100 + 400 * P_NS,
            'inclination_EM': gw_params.get('inclination', np.random.uniform(0, np.pi / 2)),
            'log10_mej_dyn': -2.0 + 0.5 * P_EMB,
            'log10_mej_wind': -1.5 + 0.5 * P_EMB,
            'KNphi': 15 + 60 * P_MassGap,
            'luminosity_distance': gw_params.get('distance', np.random.uniform(40, 150)),
            'trigger_time': np.random.uniform(-2.0, 1.0),
            'alpha': 0,
            'ratio_epsilon': 1,
            'ratio_zeta': 0.3,
            'geocent_time': 0.0,
            'phase': 0.0,
        }

        if 'chirp_mass' in gw_params:
            kn_params['chirp_mass'] = gw_params['chirp_mass']

        return kn_params

    def _write_candidate_dat_file(self, gw_params, kilonova_params, output_dir):
        """Write the BO candidate as the .dat input for nmma-create-injection."""
        dat_file = os.path.join(output_dir, "candidate.dat")

        row = {
            'simulation_id': 0,
            'longitude': float(gw_params.get('longitude', gw_params.get('ra', 0.0))),
            'latitude': float(gw_params.get('latitude', gw_params.get('dec', 0.0))),
            'inclination': float(gw_params.get(
                'inclination',
                kilonova_params.get('inclination_EM', np.random.uniform(0, np.pi))
            )),
            'distance': float(gw_params.get(
                'distance',
                kilonova_params.get('luminosity_distance', np.random.uniform(40, 150))
            )),
            'mass1': float(gw_params.get('mass1', kilonova_params.get('mass_1', 1.4))),
            'mass2': float(gw_params.get('mass2', kilonova_params.get('mass_2', 1.4))),
            'spin1z': float(gw_params.get('spin1z', 0.0)),
            'spin2z': float(gw_params.get('spin2z', 0.0)),
        }

        pd.DataFrame([row]).to_csv(dat_file, sep="\t", index=False)
        return dat_file

    def create_nmma_injection(self, gw_params, kilonova_params, output_dir):
        """Generate injection.json from candidate.dat with nmma-create-injection."""
        dat_file = self._write_candidate_dat_file(gw_params, kilonova_params, output_dir)
        injection_file = os.path.join(output_dir, "injection.json")

        cmd = self._nmma_command(
            "nmma-create-injection",
            "nmma.eos.create_injection",
        ) + [
            "--prior-file", self.prior_file,
            "--injection-file", dat_file,
            "--aligned-spin",
            "--eos-file", "./nmma/example_files/eos/ALF2.dat",
            "--binary-type", "BNS",
            "--original-parameters",
            "--extension", "json",
            "--filename", injection_file,
            "--generation-seed", "42",
        ]

        process = subprocess.Popen(
            cmd,
            cwd=os.environ.get("NMMA_WORKDIR", os.getcwd()),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._nmma_subprocess_env(),
        )
        stdout, stderr = process.communicate(timeout=60)

        if process.returncode != 0:
            raise RuntimeError(
                "nmma-create-injection failed "
                f"(return_code={process.returncode}). stderr:\n{stderr[-2000:]}"
            )

        if not os.path.exists(injection_file) or os.path.getsize(injection_file) == 0:
            raise RuntimeError(
                "nmma-create-injection did not create injection.json. "
                f"stdout tail:\n{stdout[-800:] if stdout else '(empty)'}\n"
                f"stderr tail:\n{stderr[-800:] if stderr else '(empty)'}"
            )

        return injection_file

    def _make_short_nmma_output_dir(self):
        """Create a short path for NMMA, which rejects long output directory names."""
        base_dir = "/tmp" if os.path.isdir("/tmp") else (
            os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "."
        )
        return tempfile.mkdtemp(prefix="nmma_bo_", dir=base_dir)

    def _nmma_executable(self, name):
        """Prefer NMMA executables from the active conda env over ~/.local/bin."""
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            candidate = os.path.join(conda_prefix, "bin", name)
            if os.path.exists(candidate):
                return candidate
        return shutil.which(name) or name

    def _nmma_command(self, name, module):
        """Build a command that avoids stale ~/.local NMMA console scripts."""
        exe = self._nmma_executable(name)
        local_bin = os.path.expanduser("~/.local/bin")
        if exe.startswith(local_bin) or not os.path.exists(exe):
            return [
                sys.executable,
                "-c",
                f"from {module} import main; main()",
            ]
        return [exe]

    def _copy_nmma_outputs(self, nmma_output_dir, output_dir):
        """Copy NMMA products from the short work directory to the BO output dir."""
        for src in glob.glob(os.path.join(nmma_output_dir, "*_lightcurves.png")):
            shutil.copy2(src, os.path.join(output_dir, os.path.basename(src)))

    def _nmma_subprocess_env(self):
        """Use isolated, conservative settings for NMMA subprocesses."""
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env.setdefault("MALLOC_ARENA_MAX", "1")
        env["PYTHONNOUSERSITE"] = "1"
        env.pop("PYTHONPATH", None)
        return env

    def plot_lightcurve(self, output_dir):
        """Create a combined light-curve plot from the saved lc.csv file."""
        lc_file = os.path.join(output_dir, "lc.csv")
        plot_file = os.path.join(output_dir, "lightcurve.png")

        if not os.path.exists(lc_file) or os.path.getsize(lc_file) == 0:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            df = pd.read_csv(lc_file)
            required_cols = {"filter", "mag"}
            if df.empty or not required_cols.issubset(df.columns):
                return

            time_col = "jd" if "jd" in df.columns else df.columns[0]
            if time_col not in df.columns:
                return

            colors = {
                "ztfg": "tab:green",
                "ztfr": "tab:red",
                "ztfi": "tab:purple",
                "g": "tab:green",
                "r": "tab:red",
                "i": "tab:purple",
            }

            t0 = float(df[time_col].min())
            fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)

            for filt, sub_df in df.groupby("filter"):
                sub_df = sub_df.sort_values(time_col)
                yerr = sub_df["mag_unc"] if "mag_unc" in sub_df.columns else None
                ax.errorbar(
                    sub_df[time_col] - t0,
                    sub_df["mag"],
                    yerr=yerr,
                    fmt="o-",
                    markersize=3.5,
                    linewidth=1.2,
                    capsize=2,
                    label=str(filt),
                    color=colors.get(str(filt)),
                )

            ax.invert_yaxis()
            ax.set_xlabel(f"Days since first observation ({time_col} - min({time_col}))")
            ax.set_ylabel("Magnitude")
            ax.set_title(f"NMMA light curve: {os.path.basename(output_dir)}")
            ax.grid(True, alpha=0.3)
            ax.legend(title="Filter")
            fig.tight_layout()
            fig.savefig(plot_file)
            plt.close(fig)
            print(f"      Light curve plot saved: {plot_file}")
        except Exception as exc:
            print(f"      [WARNING] Could not create light curve plot: {exc}")

    def run_kilonova_simulation(self, kilonova_params, exposure_time, output_dir, gw_params=None):
        """
        Run kilonova simulation for given parameters.

        RETURNS:
            detectable: bool - True if event is detectable by ZTF
            message: str - Status message
            peak_mag: float - Peak magnitude of the brightest filter
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            if not os.path.exists(self.svd_path):
                return False, f"SVD_PATH not found: {self.svd_path}", 99.0
            if not os.path.exists(self.prior_file):
                return False, f"PRIOR_FILE not found: {self.prior_file}", 99.0

            if gw_params is None:
                gw_params = {}

            injection_file = self.create_nmma_injection(gw_params, kilonova_params, output_dir)

            lc_frames = []
            too_frames = []
            stdout_parts = []
            stderr_parts = []
            return_codes = []

            for filt in self.filters:
                nmma_output_dir = self._make_short_nmma_output_dir()
                lc_outfile = os.path.join(nmma_output_dir, "lc.csv")

                cmd = [
                    *self._nmma_command("lightcurve-analysis", "nmma.em.analysis"),
                    "--model", "Bu2019lm",
                    "--svd-path", self.svd_path,
                    "--outdir", nmma_output_dir,
                    "--label", f"sim_{exposure_time}s_{filt}",
                    "--prior", self.prior_file,
                    "--tmin", "0.1",
                    "--tmax", "14",
                    "--dt", "0.2",
                    "--error-budget", "1",
                    "--nlive", "512",
                    "--Ebv-max", "0",
                    "--injection", injection_file,
                    "--injection-num", "0",
                    "--injection-outfile", lc_outfile,
                    "--generation-seed", "42",
                    "--filters", filt,
                    "--local-only",
                ]

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=self._nmma_subprocess_env(),
                )

                try:
                    stdout, stderr = process.communicate(timeout=100)
                    return_code = process.returncode
                except subprocess.TimeoutExpired:
                    print(f"      [WARNING] Simulation for {filt} exceeded 100s timeout — terminating.")
                    process.kill()
                    try:
                        stdout, stderr = process.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        stdout, stderr = "", "(process did not exit even after kill)"
                    return_code = -1

                return_codes.append(return_code)
                stdout_parts.append(f"[{filt}]\n{stdout or ''}")
                stderr_parts.append(f"[{filt}]\n{stderr or ''}")

                if os.path.exists(lc_outfile) and os.path.getsize(lc_outfile) > 0:
                    lc_frames.append(pd.read_csv(lc_outfile))

                too_outfile = os.path.join(nmma_output_dir, "too.csv")
                if os.path.exists(too_outfile) and os.path.getsize(too_outfile) > 0:
                    too_frames.append(pd.read_csv(too_outfile))

                self._copy_nmma_outputs(nmma_output_dir, output_dir)
                shutil.rmtree(nmma_output_dir, ignore_errors=True)

            stdout = "\n".join(stdout_parts)
            stderr = "\n".join(stderr_parts)
            return_code = 0 if any(code == 0 for code in return_codes) else (
                return_codes[-1] if return_codes else -1
            )

            if lc_frames:
                pd.concat(lc_frames, ignore_index=True).to_csv(
                    os.path.join(output_dir, "lc.csv"), index=False
                )
                self.plot_lightcurve(output_dir)
            if too_frames:
                pd.concat(too_frames, ignore_index=True).to_csv(
                    os.path.join(output_dir, "too.csv"), index=False
                )

            # Check output files
            lc_file = os.path.join(output_dir, "lc.csv")
            too_file = os.path.join(output_dir, "too.csv")

            lc_exists = os.path.exists(lc_file) and os.path.getsize(lc_file) > 0
            too_exists = os.path.exists(too_file) and os.path.getsize(too_file) > 0

            peak_mag = 99.0
            detectable = False

            if lc_exists or too_exists:
                detectable, peak_mag = self.check_detections_and_extract_peak_mag(output_dir)
            else:
                print(f"      [DEBUG] return_code={return_code}")
                print(f"      [DEBUG] stderr:\n{stderr[-2000:] if stderr else '(empty)'}")
                print(f"      [DEBUG] stdout tail:\n{stdout[-800:] if stdout else '(empty)'}")

            # Clean up temporary files
            self.cleanup_simulation_files(output_dir)

            if detectable:
                return detectable, f"DETECTABLE (peak mag: {peak_mag:.2f})", peak_mag
            elif lc_exists or too_exists:
                return detectable, f"NOT DETECTABLE (peak mag: {peak_mag:.2f})", peak_mag
            else:
                return False, "No output files created", peak_mag

        except Exception as e:
            print(f"Simulation error: {e}")
            return False, f"Error: {e}", 99.0

    def check_detections_and_extract_peak_mag(self, output_dir):
        """
        Check if simulation produced detectable events.
        A detection is: mag != 99.0 AND mag < ZTF_limit_for_that_filter
        ZTF limits: g=21.067, r=21.012, i=20.51
        """
        detectable = False
        peak_mag = 99.0

        file_path = os.path.join(output_dir, "lc.csv")
        if not os.path.exists(file_path):
            print(f"      No lc.csv file found in {output_dir}")
            return detectable, peak_mag

        try:
            df = pd.read_csv(file_path)
            
            # Filter out non-detections (mag == 99.0)
            detections = df[df['mag'] != 99.0].copy()
            
            if detections.empty:
                print(f"      All magnitudes are 99.0 (non-detections)")
                return detectable, peak_mag
            
            # Find peak magnitude (brightest = minimum magnitude)
            peak_mag = detections['mag'].min()
            
            # Get ZTF limits from detection calculator
            ztf_limits = self.detection_calc.flat_limits if self.detection_calc else {}
            print(f"      ZTF limits: {ztf_limits}")
            
            # Check each detection against ZTF limit for its filter
            for _, row in detections.iterrows():
                # Extract filter name (remove 'ztf' prefix if present)
                filt = row['filter']
                filt_key = filt.replace('ztf', '') if isinstance(filt, str) else filt
                
                # Get the ZTF limit for this filter
                limit = ztf_limits.get(filt_key)
                
                # If we have a limit and the magnitude is brighter than it
                if limit is not None and row['mag'] < limit:
                    detectable = True
                    print(f"      ✓ DETECTABLE in {filt}: mag={row['mag']:.3f} < ZTF_limit={limit:.3f}")
                    break  # One detection is enough
            
            if not detectable:
                print(f"      No detections brighter than ZTF limits. Peak mag: {peak_mag:.3f}")
                    
        except Exception as e:
            print(f"Error processing lc.csv: {e}")

        return detectable, peak_mag

    def _extract_valid_magnitudes(self, df):
        """Extract valid magnitude measurements"""
        mag_col = 'mag' if 'mag' in df.columns else 'magnitude' if 'magnitude' in df.columns else None

        if mag_col is None:
            return pd.DataFrame()

        mask = (
            (df[mag_col] < 30) &
            (df[mag_col] > 10) &
            (df[mag_col] != 99.0) &
            (~df[mag_col].isna())
        )

        valid_mags = df[mask].copy()
        return valid_mags

    def cleanup_simulation_files(self, output_dir):
        """Clean up temporary files"""
        files_to_keep = ['lc.csv', 'too.csv', 'candidate.dat', 'injection.json', 'lightcurve.png']
        # Keep plots if they exist
        plot_files = glob.glob(os.path.join(output_dir, "*_lightcurves.png"))
        files_to_keep.extend([os.path.basename(f) for f in plot_files])

        try:
            all_files = glob.glob(os.path.join(output_dir, '*'))
            for file_path in all_files:
                file_name = os.path.basename(file_path)
                if file_name not in files_to_keep and os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception:
            pass


def build_gw_params_from_X(X, feature_columns):
    """Map BO candidate vector to gw_params dict"""
    gw_params = {}
    for i, name in enumerate(feature_columns):
        if i < len(X):
            gw_params[name] = float(X[i])

    # Enforce mass1 >= mass2 convention
    if 'mass1' in gw_params and 'mass2' in gw_params:
        if gw_params['mass1'] < gw_params['mass2']:
            gw_params['mass1'], gw_params['mass2'] = gw_params['mass2'], gw_params['mass1']

    # Ensure required fields
    if 'HasNS' not in gw_params:
        gw_params['HasNS'] = 1.0
    if 'HasRemnant' not in gw_params:
        gw_params['HasRemnant'] = 1.0
    if 'HasMassGap' not in gw_params:
        gw_params['HasMassGap'] = 0.0

    return gw_params
