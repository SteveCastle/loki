import React from "react";

function HotCorner({ handleClick }) {
  return (
    <div className="hotCorner" onClick={handleClick}>
      <span></span>
      <span></span>
      <span></span>
      <span></span>
    </div>
  );
}

export default HotCorner;
