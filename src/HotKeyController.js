import React from "react";
import FocusLock from "react-focus-lock";
function Detail({ handleKeyPress }) {
  return (
    <FocusLock>
      <div
        className="hotkeyController"
        tabIndex="0"
        onKeyPress={handleKeyPress}
      />
    </FocusLock>
  );
}

export default Detail;
