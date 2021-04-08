import React, { useRef } from "react";
import useScrollOnDrag from "react-scroll-ondrag";

const url = window.require("url");
const path = window.require("path");

import { EXTENSIONS } from "./constants";

function Detail({
  fileName,
  size,
  audio,
  volume,
  videoControls,
  setAudio,
  setVolume,
  handleClick,
  handleScroll,
  controlMode,
  handleRightClick,
}) {
  const containerRef = useRef(null);
  const { events } = useScrollOnDrag(containerRef);
  return (
    <div
      className={
        controlMode.key === "MOUSE" ? "container lock-scroll" : "container"
      }
      onClick={controlMode.key === "MOUSE" ? null : handleClick}
      onContextMenu={handleRightClick}
      onWheel={controlMode.key === "MOUSE" ? handleScroll : null}
      tabIndex="0"
      {...events}
      ref={controlMode.key === "MOUSE" ? containerRef : null}
    >
      {EXTENSIONS.img.includes(path.extname(fileName).toLowerCase()) && (
        <img
          key={fileName}
          src={url.format({
            protocol: "file",
            pathname: fileName,
          })}
          className={size.className}
        />
      )}
      {EXTENSIONS.video.includes(path.extname(fileName).toLowerCase()) && (
        <video
          className={size.className}
          src={url.format({
            protocol: "file",
            pathname: fileName,
          })}
          loop
          autoPlay
          controls={videoControls}
          controlsList="nofullscreen nodownload noremoteplayback"
          muted={!audio}
          onLoadStart={(e) => (e.target.volume = volume)}
          onVolumeChange={(e) => {
            setVolume(e.target.volume);
            setAudio(!e.target.muted);
          }}
        />
      )}
    </div>
  );
}

export default Detail;
