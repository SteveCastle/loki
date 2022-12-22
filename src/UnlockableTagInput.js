import React, { useState } from "react";
import FocusLock from "react-focus-lock";
import lock from "./assets/lock-fill.svg";

import "./UnlockableTagInput.css";

export default function UnlockableTagInput({ activeTag, setActiveTag }) {
  const [isLocked, setIsLocked] = useState(true);
  return (
    <div className="UnlockableTagInput">
      {isLocked ? (
        <div className="inputGroup">
          <input
            type="text"
            value={activeTag.category}
            onChange={(e) =>
              setActiveTag({
                category: e.target.value,
                tag: activeTag.tag,
              })
            }
          />
          <input
            type="text"
            value={activeTag.tag}
            onChange={(e) =>
              setActiveTag({
                tag: e.target.value,
                category: activeTag.category,
              })
            }
          />
        </div>
      ) : (
        <div className="inputGroup">
          <FocusLock>
            <input
              type="text"
              value={activeTag.category}
              onChange={(e) =>
                setActiveTag({
                  category: e.target.value,
                  tag: activeTag.tag,
                })
              }
            />
            <input
              type="text"
              value={activeTag.tag}
              onChange={(e) =>
                setActiveTag({
                  tag: e.target.value,
                  category: activeTag.category,
                })
              }
            />
          </FocusLock>
        </div>
      )}
      <button
        className="saveSettingsButton"
        onClick={() => setIsLocked(!isLocked)}
        style={{ opacity: isLocked ? 1 : 0.6 }}
      >
        <img src={lock} />
      </button>
    </div>
  );
}
