const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL; 

export async function startOptimizeJob(payload: {
    job_description: string;
    resume_text: string;
    resume_format: string; 
}){

    const res = await fetch(`${API_BASE}/optimize/async`,{
        method: 'POST',
        headers: {
            "Content-Type": "application/json",
            "Idempotency-Key": crypto.randomUUID(),
        },
        body: JSON.stringify(payload),
    });

    if(!res.ok){
        throw new Error("Failed to start optimization job");
    }

    return res.json();
}

export async function getJobStatus(jobId: string){
    const res = await fetch(`${API_BASE}/optimize/status/${jobId}`);

    if(!res.ok){
        throw new Error("Failed to fetch job status");
    }

    return res.json();
}

// export async function uploadResumeFile(file: File) {
//     const formData = new FormData();
//     formData.append("file", file);
  
//     const res = await fetch(
//       `${process.env.NEXT_PUBLIC_API_URL}/optimize/upload`,
//       {
//         method: "POST",
//         body: formData,
//       }
//     );
  
//     if (!res.ok) {
//       let message = "Upload failed";
//       try {
//         const err = await res.json();
//         message = err.detail || message;
//       } catch {}
//       throw new Error(message);
//     }
  
//     return res.json() as Promise<{
//       filename: string;
//       text: string;
//     }>;
//   }

export  async function uploadResumeFile(file: File) {
      const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL;
    
      if (!apiBase) {
        throw new Error("API URL not configured");
      }
    
      const formData = new FormData();
      formData.append("file", file);
    
      const res = await fetch(`${apiBase}/optimize/upload`, {
        method: "POST",
        body: formData,
      });
    
      if (!res.ok) {
        let message = "Upload failed";
        try {
          const err = await res.json();
          message = err.detail ?? message;
        } catch {}
        throw new Error(message);
      }
    
      return res.json();
    }
    
  
export async function fetchJDFromUrl(url: string) {
    const res = await fetch(
      `${process.env.NEXT_PUBLIC_API_BASE_URL}/optimize/jd-from-url`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url }),
      }
    );
  
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to extract JD");
    }
  
    return res.json();
  }
  