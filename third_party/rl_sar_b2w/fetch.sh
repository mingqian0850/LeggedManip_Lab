#!/usr/bin/env bash
# Download the pinned public rl_sar B2W policy and its deployment metadata.
# Upstream: https://github.com/fan-ziqi/rl_sar
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly REVISION="376d42c9b128f963ab08579762d5a216a976ce39"
readonly RAW_BASE="https://raw.githubusercontent.com/fan-ziqi/rl_sar/${REVISION}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
artifact_dir="${script_dir}/artifacts"
mkdir -p -- "${artifact_dir}"
download_dir="$(mktemp -d "${artifact_dir}/.download.XXXXXX")"
trap 'rm -rf -- "${download_dir}"' EXIT

download() {
    local source_path="$1"
    local output_name="$2"
    local output_path="${download_dir}/${output_name}"

    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --silent --show-error \
            --output "${output_path}" "${RAW_BASE}/${source_path}"
    elif command -v wget >/dev/null 2>&1; then
        wget --quiet --output-document="${output_path}" "${RAW_BASE}/${source_path}"
    else
        echo "error: curl or wget is required" >&2
        exit 1
    fi

    local expected
    expected="$(awk -v path="artifacts/${output_name}" '$2 == path { print $1 }' "${script_dir}/SHA256SUMS")"
    if [[ -z "${expected}" ]]; then
        echo "error: no checksum recorded for ${output_name}" >&2
        exit 1
    fi
    echo "${expected}  ${output_path}" | sha256sum --check --status
}

download "policy/b2w/robot_lab/policy.pt" "policy.pt"
download "policy/b2w/robot_lab/config.yaml" "config.yaml"
download "policy/b2w/base.yaml" "base.yaml"
download "LICENSE" "LICENSE"

for name in policy.pt config.yaml base.yaml LICENSE; do
    mv -- "${download_dir}/${name}" "${artifact_dir}/${name}"
done

(cd -- "${script_dir}" && sha256sum --check SHA256SUMS)
echo "Downloaded pinned rl_sar B2W artifacts to ${artifact_dir}"
