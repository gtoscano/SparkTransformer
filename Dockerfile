FROM nvcr.io/nvidia/pytorch:25.09-py3

ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=gtoscano
ARG UID=1000
ARG GID=1000
ARG HOME_DIR=/home/${USERNAME}

# ------------------------------------------------------------------------------
# System tools (no pipenv)
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      sudo \
      git \
      ca-certificates \
      bash \
      tini \
  && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# Robust user/group creation + sudo setup
# ------------------------------------------------------------------------------
RUN set -eux; \
  # Group creation / remapping
  if getent group "${GID}" >/dev/null; then \
    existing_group="$(getent group "${GID}" | cut -d: -f1)"; \
    if [ "${existing_group}" != "${USERNAME}" ]; then \
      groupmod -n "${USERNAME}" "${existing_group}"; \
    fi; \
  else \
    groupadd -g "${GID}" "${USERNAME}"; \
  fi; \
  \
  # User creation / remapping
  if id -u "${USERNAME}" >/dev/null 2>&1; then \
    usermod -u "${UID}" -g "${GID}" -d "${HOME_DIR}" -s /bin/bash "${USERNAME}"; \
  elif getent passwd "${UID}" >/dev/null; then \
    prev_user="$(getent passwd "${UID}" | cut -d: -f1)"; \
    usermod -l "${USERNAME}" -d "${HOME_DIR}" -m "${prev_user}"; \
    usermod -g "${GID}" -s /bin/bash "${USERNAME}"; \
  else \
    useradd -m -u "${UID}" -g "${GID}" -s /bin/bash "${USERNAME}"; \
  fi; \
  \
  # Passwordless sudo for this user (handy for debugging)
  usermod -aG sudo "${USERNAME}"; \
  echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}; \
  chmod 0440 /etc/sudoers.d/${USERNAME}

# ------------------------------------------------------------------------------
# Python dependencies (no pipenv, keep NGC torch)
#   --no-deps so we don't overwrite CUDA-enabled torch from the base image
# ------------------------------------------------------------------------------
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --no-deps \
      transformers \
      peft \
      datasets \
      "trl==0.19.1" \
      "bitsandbytes==0.48" \
      accelerate \
      "huggingface_hub[cli]>=0.34.0,<1.0" \
      "tokenizers>=0.22.0,<0.24.0" \
      hf_transfer \
      hf_xet \
      xxhash \
      multiprocess \
      onnxruntime \
      onnx \
      onnxruntime-tools \
      optimum-onnx \
      pandas \
      pytz \
      pyarrow
     

# If you want vLLM in this same image, uncomment:
# RUN pip install --no-cache-dir --no-deps vllm

# ------------------------------------------------------------------------------
# Hugging Face cache + runtime env
# (path matches the non-root user; feel free to adjust)
# ------------------------------------------------------------------------------
ENV HOME=${HOME_DIR} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=${HOME_DIR}/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=${HOME_DIR}/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN mkdir -p "${HF_HOME}" && chown -R "${UID}:${GID}" "${HOME_DIR}"

WORKDIR /workspace

# ------------------------------------------------------------------------------
# Switch to non-root user
# ------------------------------------------------------------------------------
USER ${USERNAME}

# Simple entrypoint, like the NGC docs (no pipenv logic)
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]

