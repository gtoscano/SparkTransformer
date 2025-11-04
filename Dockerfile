FROM nvcr.io/nvidia/pytorch:25.09-py3

ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=gtoscano
ARG UID=1000
ARG GID=1000
ARG HOME_DIR=/home/${USERNAME}

# Tools
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates bash tini \
  && rm -rf /var/lib/apt/lists/*

# Robust user/group creation (handles existing UID/GID)
RUN set -eux; \
  if getent group "${GID}" >/dev/null; then \
    existing_group="$(getent group "${GID}" | cut -d: -f1)"; \
    if [ "${existing_group}" != "${USERNAME}" ]; then groupmod -n "${USERNAME}" "${existing_group}"; fi; \
  else \
    groupadd -g "${GID}" "${USERNAME}"; \
  fi; \
  if id -u "${USERNAME}" >/dev/null 2>&1; then \
    usermod -u "${UID}" -g "${GID}" -d "${HOME_DIR}" -s /bin/bash "${USERNAME}"; \
  elif getent passwd "${UID}" >/dev/null; then \
    prev_user="$(getent passwd "${UID}" | cut -d: -f1)"; \
    usermod -l "${USERNAME}" -d "${HOME_DIR}" -m "${prev_user}"; \
    usermod -g "${GID}" -s /bin/bash "${USERNAME}"; \
  else \
    useradd -m -u "${UID}" -g "${GID}" -s /bin/bash "${USERNAME}"; \
  fi
# mkdir -p "${HOME_DIR}/.cache/huggingface" && chown -R "${UID}:${GID}" "${HOME_DIR}"

# Python deps
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      "transformers" \
      "peft" \
      "datasets" \
      "trl==0.19.1" \
      "bitsandbytes==0.48" \
      "accelerate" \
      "huggingface_hub[cli]"

# Runtime env
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=${HOME_DIR}

#ENV HF_HOME=${HOME_DIR}/.cache/huggingface \
#    HUGGINGFACE_HUB_CACHE=${HOME_DIR}/.cache/huggingface \
#    HF_HUB_ENABLE_HF_TRANSFER=1 \
#    PIP_DISABLE_PIP_VERSION_CHECK=1 \
#    HOME=${HOME_DIR}
WORKDIR /workspace
USER ${USERNAME}

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]

