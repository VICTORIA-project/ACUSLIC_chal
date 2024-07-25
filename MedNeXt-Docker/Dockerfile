FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Copy nnU-Net results folder into resources folder
# The required files for nnUNet inference are:
# resources/nnUNet_results/.../
# |-- plans.json
# |-- dataset.json
# |-- fold_0/
# |---- checkpoint_final.pth
# |-- fold_1/
# |---- checkpoint_final.pth
# |-- fold_2/
# |---- checkpoint_final.pth
# |-- fold_3/
# |---- checkpoint_final.pth
# |-- fold_4/
# |---- checkpoint_final.pth
COPY --chown=user:user resources /opt/app/resources

# Copy the setup.py file and nnunetv2 folder to the container
COPY --chown=user:user setup.py /opt/app/
COPY --chown=user:user nnunetv2 /opt/app/nnunetv2

# Install the application and its dependencies using setup.py
RUN python -m pip install --user --no-cache-dir /opt/app

# Copy the inference script, the postprocessing script and utils to the container
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user postprocess_probability_maps.py /opt/app/
COPY --chown=user:user model.py /opt/app/
COPY --chown=user:user test_run.sh /opt/app/

# Ensure the script has executable permissions
RUN chmod +x /opt/app/test_run.sh

ENTRYPOINT ["python", "inference.py"]