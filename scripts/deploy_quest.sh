#!/usr/bin/env bash
# Install the viewer on a USB-connected Quest, push a splat, start it, and show
# the app's [Splat] log lines.
#
#   ./deploy_quest.sh                                # install APK + launch
#   ./deploy_quest.sh work/quest.ply                 # install APK + push + launch
#   NO_INSTALL=1 ./deploy_quest.sh other.ply         # new capture, same APK
#
# The headset needs developer mode on and this computer's USB debugging
# prompt accepted inside the headset. The APK is built by
# unity/QuestSplatViewer/Assets/Editor/QuestProjectSetup.cs (BuildApk).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLY="${1:-}"   # optional: with splat_server.py the app fetches captures itself
APK="${2:-$HERE/../unity/QuestSplatViewer/Builds/QuestSplatViewer.apk}"
PKG="${PKG:-com.vrproj.questsplatviewer}"
# The app opens the newest splat file, so a fresh push is what it shows.
REMOTE_NAME="${REMOTE_NAME:-quest.ply}"

# adb and aapt ship with Unity's Android module when they aren't on PATH.
SDK=$(ls -d "$HOME"/Unity/Hub/Editor/*/Editor/Data/PlaybackEngines/AndroidPlayer/SDK 2>/dev/null | tail -1 || true)
ADB="${ADB:-$(command -v adb || echo "$SDK/platform-tools/adb")}"
AAPT="${AAPT:-$(command -v aapt || ls "$SDK"/build-tools/*/aapt 2>/dev/null | tail -1)}"

[[ -z "$PLY" || -f "$PLY" ]] || { echo "no such splat file: $PLY" >&2; exit 1; }
state=$("$ADB" get-state 2>&1 || true)
if [[ "$state" != "device" ]]; then
    echo "no headset ready over adb ($state)." >&2
    echo "Plug the Quest in, enable developer mode, and accept the USB debugging" \
         "prompt inside the headset; 'no permissions' means a udev rule is missing." >&2
    "$ADB" devices -l >&2
    exit 1
fi

if [[ -z "${NO_INSTALL:-}" ]]; then
    [[ -f "$APK" ]] || { echo "no APK at $APK -- build it first" >&2; exit 1; }
    echo "=== install $(basename "$APK") (can take a few minutes on a Quest) ==="
    # Streamed install fails on Quest with an empty error; copy first instead.
    "$ADB" install --no-streaming -r "$APK"
fi

# Application.persistentDataPath: adb can write here without the app holding
# any storage permission.
if [[ -n "$PLY" ]]; then
    DEST="/sdcard/Android/data/$PKG/files"
    echo "=== push $(basename "$PLY") -> $DEST/$REMOTE_NAME ==="
    "$ADB" shell mkdir -p "$DEST"
    "$ADB" push "$PLY" "$DEST/$REMOTE_NAME"
    "$ADB" shell touch "$DEST/$REMOTE_NAME"   # push keeps the source mtime; the app opens the newest
fi

ACTIVITY=""
if [[ -n "$AAPT" && -f "$APK" ]]; then
    ACTIVITY=$("$AAPT" dump badging "$APK" | sed -n "s/^launchable-activity: name='\([^']*\)'.*/\1/p" | head -1)
fi
echo "=== launch $PKG ==="
"$ADB" shell am force-stop "$PKG"
"$ADB" logcat -c
if [[ -n "$ACTIVITY" ]]; then
    "$ADB" shell am start -n "$PKG/$ACTIVITY"
else
    "$ADB" shell monkey -p "$PKG" -c com.oculus.intent.category.VR 1
fi

echo "=== app log (Ctrl+C to stop) ==="
"$ADB" logcat -s Unity:V | grep --line-buffered -E "\[Splat\]|Exception|Error"
