import React, { useRef } from "react";
import useScrollOnDrag from "react-scroll-ondrag";

const url = window.require("url");
const path = window.require("path");

import { SIZE, EXTENSIONS, VIEW } from "./constants";

function Detail({ fileName, size, handleClick, handleScroll, controlMode }) {
  const containerRef = useRef(null);
  const { events } = useScrollOnDrag(containerRef);
  return (
    <div
      className="container"
      onClick={controlMode === "MOUSE" ? null : handleClick}
      onWheel={controlMode === "MOUSE" ? handleScroll : null}
      tabIndex="0"
      {...events}
      ref={controlMode === "MOUSE" ? containerRef : null}
    >
      {EXTENSIONS.img.includes(path.extname(fileName).toLowerCase()) && (
        <img
          key={fileName}
          src={url.format({
            protocol: "file",
            pathname: fileName
          })}
          className={`${size === SIZE.OVERSCAN ? "overscan" : null}`}
        />
      )}
      {EXTENSIONS.video.includes(path.extname(fileName).toLowerCase()) && (
        <video
          className={`${size === SIZE.OVERSCAN ? "overscan" : null}`}
          src={url.format({
            protocol: "file",
            pathname: fileName
          })}
          loop
          autoPlay
          muted
          controls
        />
      )}
    </div>
  );
}

export default Detail;
