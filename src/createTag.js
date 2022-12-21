const exec = window.require("child_process").exec;

const createTag = (category, tag, filePath) => {
  const command = `ix tag ${category} ${tag} "${filePath}"`;
  // const command = `copy "${filePath}" G:\STier`;

  exec(command, (error, stdout, stderr) => {
    if (error) {
      console.error(`exec error: ${error}`);
      return;
    }
    console.log(`stdout: ${stdout}`);
    console.log(`stderr: ${stderr}`);
  });
};

export default createTag;
