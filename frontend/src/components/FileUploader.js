import React from "react";

export default function FileUploader({ onDetect, setTableData }) {
  const handle = async e => {
    const file = e.target.files[0];
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    onDetect(data.columns);

    // read data for preview if you like
    const text = await file.text();
    const rows = text
      .split("\n")
      .slice(1)
      .map(r => r.split(","));
    setTableData(rows);
  };

  return <input type="file" onChange={handle} />;
}
