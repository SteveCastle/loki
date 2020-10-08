import React, { useState } from "react";
import FocusLock from "react-focus-lock";

function Tags({ onHide, tags = [] }) {
  const [value, setValue] = useState("");
  const handleChange = (e) => {
    console.log(e);
    setValue(e.target.value);
  };
  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      onHide();
      console.log("Entered");
    }
    if (e.key === "Escape") {
      onHide();
      console.log("Escape");
    }
  };

  return (
    <div className="tags">
      <ul>
        {tags.map((tag) => (
          <li>{tag}</li>
        ))}
      </ul>
      <FocusLock>
        <input
          onChange={handleChange}
          onKeyDown={handleKeyPress}
          value={value}
        ></input>
      </FocusLock>
    </div>
  );
}

export default Tags;
