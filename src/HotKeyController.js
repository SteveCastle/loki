import React, { useRef, useEffect } from "react";
import FocusLock from "react-focus-lock";
function Detail({ handleKeyPress }) {
  const ref = useRef(null);

  useEffect(() => {
    ref.current.focus();
  });

  return (
    <FocusLock>
      <div
        className="hotkeyController"
        tabIndex="0"
        onKeyPress={handleKeyPress}
        ref={ref}
      />
    </FocusLock>
  );
}

export default Detail;
