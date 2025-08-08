import React, { useState } from "react";
import axios from "axios";

export default function Upload() {
  const [file,setFile]=useState(),[info,setInfo]=useState();
  const upload=async()=>{
    const fd=new FormData(); fd.append("file",file);
    const res=await axios.post("/api/upload", fd);
    setInfo(res.data);
  };
  return (
    <div className="mb-4">
      <h2 className="font-semibold mb-1">Upload CSV</h2>
      <input type="file" onChange={e=>setFile(e.target.files[0])} />
      <button className="ml-2 px-4 py-1 bg-blue-500 text-white rounded" onClick={upload}>
        Upload
      </button>
      {info && <pre className="mt-2 bg-gray-100 p-2 rounded">{JSON.stringify(info, null, 2)}</pre>}
    </div>
  );
}
