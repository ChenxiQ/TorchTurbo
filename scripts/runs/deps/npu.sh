#!/bin/bash

if ! pip show mindspeed 2>/dev/null | grep -qx 'Version: 0.15.3'; then
    rm -rf /opt/MindSpeed
    git clone --branch core_r0.15.3 \
        https://gitcode.com/Ascend/MindSpeed.git \
        /opt/MindSpeed
    pip install -e /opt/MindSpeed
fi
