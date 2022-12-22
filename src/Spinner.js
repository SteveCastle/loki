import styled from "styled-components";

import React from "react";

const StyledSpinner = styled.div`
  width: 100%;
  display: flex;
  align-items: flex-end;
  justify-content: flex-end;
  position: relative;
  height: 100%;
  top: 0;
  left: 0;
  opacity: 0.7;
  z-index: 999;

  & .spinner,
  & .spinner:after {
    border-radius: 50%;
    width: 2em;
    height: 2em;
  }
  & .spinner {
    font-size: 10px;
    position: relative;
    text-indent: -9999em;
    border-top: 1.1em solid rgba(1, 1, 1, 0.2);
    border-right: 1.1em solid rgba(1, 1, 1, 0.2);
    border-bottom: 1.1em solid rgba(1, 0, 1, 0.2);
    border-left: 1.1em solid #fff;
    -webkit-transform: translateZ(0);
    -ms-transform: translateZ(0);
    transform: translateZ(0);
    -webkit-animation: load8 1.1s infinite linear;
    animation: load8 1.1s infinite linear;
  }
  @-webkit-keyframes load8 {
    0% {
      -webkit-transform: rotate(0deg);
      transform: rotate(0deg);
    }
    100% {
      -webkit-transform: rotate(360deg);
      transform: rotate(360deg);
    }
  }
  @keyframes load8 {
    0% {
      -webkit-transform: rotate(0deg);
      transform: rotate(0deg);
    }
    100% {
      -webkit-transform: rotate(360deg);
      transform: rotate(360deg);
    }
  }
`;

const Spinner = () => (
  <StyledSpinner>
    <div className="spinner">Loading</div>
  </StyledSpinner>
);

export default Spinner;
