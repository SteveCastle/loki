import React from "react";
import FocusLock from "react-focus-lock";
const settings = window.require("electron-settings");
import { HOT_KEY_DEFAULTS } from "./constants";

import "./HotKeySetter.css";
export function HotKeySetter({ action, handleComplete, setHotKeys }) {
  return (
    <FocusLock>
      <div
        tabIndex="0"
        className="HotKeySetter"
        onKeyDown={(e) => {
          console.log(e.key);
          if (!settings.has("settings.hotKeys")) {
            settings.set("settings.hotKeys", HOT_KEY_DEFAULTS);
          }
          const hotKeys = settings.get("settings.hotKeys");
          const hotKeysWithoutCurrentAction = Object.entries(hotKeys).reduce(
            (acc, [key, value]) =>
              value !== action ? { ...acc, [key]: value } : acc,
            {}
          );
          setHotKeys({ ...hotKeysWithoutCurrentAction, [e.key]: action });
          settings.set("settings.hotKeys", {
            ...hotKeysWithoutCurrentAction,
            [e.key]: action,
          });
          handleComplete();
        }}
      >
        <span>Press any key to set a new hotkey for {action}</span>
      </div>
    </FocusLock>
  );
}
