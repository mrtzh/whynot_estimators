"""Module level installation script."""
import os
import subprocess
import sys

import click

# Map of estimator name to python, CRAN, and R dependencies on github.
ESTIMATORS = {
    "causal_bart": {
        "python": ["rpy2==3.1.0"],
        "cran": ["dbarts"],
        "github_r": ["vdorie/bartCause"],
    },
    "causal_forest": {"python": ["rpy2==3.1.0"], "cran": ["grf"],},
    "deconfounder": {"python": ["tensorflow==1.*", "tensorflow_probability<=0.8"],},
    "doubleml": {"python": ["econml"],},
    "ip_weighting": {"python": ["rpy2==3.1.0"], "cran": ["WeightIt", "survey"],},
    "matching": {"python": ["rpy2==3.1.0"], "cran": ["Matching"],},
    "rlearner": {"python": ["causalml"],},
    "slearner": {"python": ["causalml"],},
    "tmle": {"python": ["rpy2==3.1.0"], "cran": ["tmle", "dbarts", "gam"]},
}


@click.group()
def main():
    """Entry point for CLI."""


def install_python(package):
    """Install a python package using PIP."""
    if package == "rpy2" and not r_build_found():
        print("Cannot install rpy2!")
        raise ValueError("Unable to find R installation. Please install R and retry.")

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", package]
    return subprocess.check_call(cmd, env=os.environ.copy())


def r_build_found():
    """Return whether or not an R build is accessible.

    Works by trying to find R's home directory (aka R_HOME).
    First it check for an environment variable R_HOME, and
    and if none is found it tries to get it from an R executable
    in the PATH.

    See: https://github.com/rpy2/rpy2/blob/master/rpy2/situation.py.
    """
    if os.environ.get("R_HOME"):
        return True
    try:
        _ = subprocess.check_output(("R", "RHOME"), universal_newlines=True)
    # pylint:disable-msg=broad-except
    except Exception:  # FileNotFoundError, WindowsError, etc
        return False
    return True


def install_r_package(package, location="CRAN"):
    """Install R packages from CRAN.

    Parameters
    ----------
        package: str
            Name of the package to install.
        location: str
            Either CRAN or GITHUB

    """
    if location not in ["CRAN", "GITHUB"]:
        raise ValueError(f"location {location} must be either CRAN or GITHUB.")

    # Ensure R and rpy2 are accessible.
    if not r_build_found():
        msg = (
            "Unable to find R installation. Please install R and retry.\n"
            "Installation instructions available at: "
            "https://github.com/zykls/whynot_estimators"
        )
        raise ValueError(msg)
    try:
        import rpy2.robjects.packages as rpackages
    except ImportError:
        raise ValueError(
            "Unable to import rpy2. rpy2 is required to install R packages."
        )

    if rpackages.isinstalled(package):
        return

    # pylint:disable-msg=no-member
    utils = rpackages.importr("utils")
    utils.chooseCRANmirror(ind=1)
    if location == "CRAN":
        utils.install_packages(package)
    else:
        if not rpackages.isinstalled("remotes"):
            utils.install_packages("remotes")
        remotes = rpackages.importr("remotes")
        remotes.install_github(package)


@main.command("show_all")
def show_all():
    """Show all of the estimators available in whynot_estimators."""
    print("Name \t\t\t\t R Required?")
    for estimator, dependencies in ESTIMATORS.items():
        requires_r = False
        if "cran" in dependencies and dependencies["cran"]:
            requires_r = True
        elif "github_r" in dependencies and dependencies["github_r"]:
            requires_r = True

        print(f"{estimator} \t\t\t {requires_r}")


@main.command("install")
@click.argument("estimator", type=str)
def install(estimator):
    """Install an estimator in whynot."""
    if estimator not in ESTIMATORS and estimator != "all":
        all_estimators = "\n\t".join(ESTIMATORS.keys())
        msg = f"Estimator {estimator} not currently supported.\n Available:\n\t{all_estimators}"
        raise ValueError(msg)

    def run_install(estimator_name):
        dependencies = ESTIMATORS[estimator_name]
        if "python" in dependencies:
            for package in dependencies["python"]:
                install_python(package)
        if "cran" in dependencies:
            for package in dependencies["cran"]:
                install_r_package(package, location="CRAN")
        if "github_r" in dependencies:
            for package in dependencies["github_r"]:
                install_r_package(package, location="GITHUB")

    if estimator == "all":
        for estimator_name in ESTIMATORS:
            try:
                run_install(estimator_name)
            # pylint:disable-msg=broad-except
            except Exception:
                print("=" * 80)
                print(f"Failed to install estimator {estimator_name}!")
                print("=" * 80)
                continue
    else:
        run_install(estimator)


if __name__ == "__main__":
    main()
