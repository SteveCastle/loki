import React, { Component } from "react";
import { FixedSizeGrid as Grid } from "react-window";
const url = require("url");

const Cell = ({ data, columnIndex, rowIndex, style }) =>
  data.fileList[rowIndex * data.columns + columnIndex] ? (
    <div
      style={{
        ...style,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        overflow: "hidden"
      }}
    >
      {(
        (data.fileList[rowIndex * data.columns + columnIndex] &&
          data.fileList[rowIndex * data.columns + columnIndex].fileName) ||
        ""
      ).includes(".webm") ||
      (
        (data.fileList[rowIndex * data.columns + columnIndex] &&
          data.fileList[rowIndex * data.columns + columnIndex].fileName) ||
        ""
      ).includes(".mp4") ? (
        //   <video
        //     className={styles.listVideo}
        //     src={data.fileList[rowIndex * data.columns + columnIndex].fileName}
        //     loop
        //     autoPlay
        //     muted
        //   />
        <video
          onClick={() =>
            data.handleClick(rowIndex * data.columns + columnIndex)
          }
          style={{
            cursor: "pointer"
          }}
          className="listImage"
          src={
            data.fileList[rowIndex * data.columns + columnIndex] &&
            url.format({
              protocol: "file",
              pathname:
                data.fileList[rowIndex * data.columns + columnIndex].fileName
            })
          }
          loop
          muted
          autoPlay
        />
      ) : (
        <img
          className="listImage"
          onClick={() =>
            data.handleClick(rowIndex * data.columns + columnIndex)
          }
          src={
            data.fileList[rowIndex * data.columns + columnIndex] &&
            url.format({
              protocol: "file",
              pathname:
                data.fileList[rowIndex * data.columns + columnIndex].fileName
            })
          }
        />
      )}
    </div>
  ) : (
    <div></div>
  );

export default class List extends Component {
  gridRef = null;
  rows = this.props.tall
    ? Math.ceil(this.props.fileList.length / 3)
    : Math.ceil(Math.sqrt(this.props.fileList.length));
  columns = this.props.tall
    ? 3
    : Math.ceil(Math.sqrt(this.props.fileList.length));
  componentDidMount() {
    this.gridRef.scrollToItem({
      columnIndex: this.props.cursor % this.columns,

      rowIndex: Math.floor(this.props.cursor / this.columns)
    });
  }
  render() {
    const { fileList, handleClick } = this.props;
    return (
      <div className="container" data-tid="container">
        <Grid
          ref={r => {
            this.gridRef = r;
          }}
          columnCount={this.columns}
          columnWidth={window.innerWidth / 3}
          height={window.innerHeight}
          rowCount={this.rows}
          rowHeight={600}
          width={window.innerWidth}
          itemData={{ fileList, handleClick, columns: this.columns }}
        >
          {Cell}
        </Grid>
      </div>
    );
  }
}
