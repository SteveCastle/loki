import React from "react";
import FocusLock from "react-focus-lock";
function HotKeyController({ handleKeyPress, showingTags }) {
  return !showingTags ? (
    <FocusLock>
      <div
        className="hotkeyController"
        tabIndex="0"
        onKeyDown={handleKeyPress}
      />
    </FocusLock>
  ) : null;
}

export default HotKeyController;
