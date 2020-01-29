import React from "react";

function HotCorner({ handleClick }) {
  return (
    <div className="hotCorner" onClick={handleClick}>
      📂
    </div>
  );
}

export default HotCorner;
