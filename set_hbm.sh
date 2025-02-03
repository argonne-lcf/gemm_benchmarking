#!/usr/bin/env bash

numactl -m $((PALS_LOCAL_RANKID + 2)) "$@"
