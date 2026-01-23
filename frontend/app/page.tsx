"use client";

import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown"
import { optimizeResumeAsync, getJobStatus} from "@/lib/api";
import type { JobStatusResponse } from "@/lib/api";
import toast from "react-hot-toast";
import mammoth from "mammoth";
import dynamic from "next/dynamic"
import { uploadResumeFile, fetchJDFromUrl } from "./lib/api";


export default function Home() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatusResponse | null>(null);
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<"resume" | "cover">("resume")
  const [progress, setProgress] = useState(0)
  const [jobDescription, setJobDescription] = useState("")
  const [resumeText, setResumeText] = useState("")
  const [resumeMode, setResumeMode] = useState<"paste" | "upload">("paste")
  const [jdMode, setJdMode] = useState<"paste" | "url">("paste")
  const [jdUrl, setJdUrl] = useState("")
  const [uploading, setUploading] = useState(false);

  // job status progress
  useEffect(() => {
    if (!jobStatus) {
      setProgress(0)
      return
    }
  
    switch (jobStatus.status) {
      case "pending":
        setProgress(10)
        break
      case "running":
        setProgress(60)
        break
      case "success":
        setProgress(100)
        break
      case "failed":
        setProgress(100)
        break
    }
  }, [jobStatus])

  // Restore page on load
  useEffect(() => {
    const savedJobId = localStorage.getItem("jobId")
    if (savedJobId) {
      toast("Previous job expired. Please start a new one.", { icon: "ℹ️" });
      localStorage.removeItem("jobId");
    }
    
  }, [])
  
  // Copy to clipboard
  const copyToClipboard = async (text: string) => {
    await navigator.clipboard.writeText(text)
    toast.success("Copied to clipboard")
  }

  // Download
  const download = (content: string, filename: string) => {
    const blob = new Blob([content], { type: "text/markdown" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }
  
  const startJob = async () => { 
    setLoading(true)
    setError(null)
    setJobStatus(null)
    setProgress(0)
    setJobId(null)
    setActiveTab("resume")

    try {

      // const res = await optimizeResumeAsync(
      //   {
      //     job_description: "Seeking a data scientist with strong Python, SQL, and machine learning skills. Must have experience with pandas, scikit-learn, and data visualization libraries. Knowledge of statistical analysis and A/B testing required.",
        
      //     resume_text: "Data scientist skilled in Python, SQL, and machine learning using pandas and scikit-learn. Experienced in statistical analysis, data visualization, and designing A/B testing frameworks.",
        
      //     resume_format: "markdown"
      //   },
      //   crypto.randomUUID()
      // )

      const res = await optimizeResumeAsync(
        {
          job_description: jobDescription,
          resume_text: resumeText,
          resume_format: "markdown"
        },
        crypto.randomUUID()
      )

      setJobId(res.job_id)
      toast.success("Optimization started")
      localStorage.setItem("jobId", res.job_id)
    } catch (e) {
      toast.error("Failed to start optimization")      
      setLoading(false)
    }
  }
  
  useEffect(() => {
    if (!jobId) return
    const interval = setInterval(async () => {
      try {
        const status = await getJobStatus(jobId)
        setJobStatus(status)

        if (status.status === "success" || status.status === "failed") {
          clearInterval(interval)
          setLoading(false)
          localStorage.removeItem("jobId")
        }
      } catch (e: any) {
        clearInterval(interval)
        if (e.message?.includes("404")) {
          setError("Job expired. Please submit again.")
          setJobId(null)
          localStorage.removeItem("jobId")
        } else {
          toast.error("Failed to fetch job status")

        }
        setLoading(false)
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [jobId])

  useEffect(() => {
    if (jobStatus?.status === "success") {
      toast.success("Optimization completed")
      window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" })
    }
  }, [jobStatus?.status])
  
  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100">
      <div className="mx-auto max-w-7xl px-6 py-10">

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-semibold tracking-tight">
            Resume Optimizer
          </h1>
          <p className="mt-2 text-neutral-400 max-w-xl">
            ATS-optimized resumes and cover letters generated with AI.
          </p>
        </div>

        {/* Bento Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Job Description */}
        <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-4">
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm font-medium text-neutral-300">
              Job Description
            </label>

            <div className="flex gap-2 text-xs">
              <button
                onClick={() => setJdMode("paste")}
                className={`px-2 py-1 rounded ${
                  jdMode === "paste"
                    ? "bg-blue-600 text-white"
                    : "text-neutral-400 hover:text-neutral-200"
                }`}
              >
                Paste
              </button>
              <button
                onClick={() => setJdMode("url")}
                className={`px-2 py-1 rounded ${
                  jdMode === "url"
                    ? "bg-blue-600 text-white"
                    : "text-neutral-400 hover:text-neutral-200"
                }`}
              >
                URL
              </button>
            </div>
          </div>

          {jdMode === "paste" ? (
            <>
              <textarea
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
                rows={6}
                placeholder="Paste the job description here…"
                className="w-full resize-none rounded-lg bg-neutral-900 border border-neutral-800 p-3 text-sm text-neutral-100 placeholder-neutral-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <div className="mt-1 text-xs text-neutral-500 text-right">
                {jobDescription.length} characters
              </div>
            </>
          ) : (
            <>
              <input
                type="url"
                value={jdUrl}
                onChange={(e) => setJdUrl(e.target.value)}
                placeholder="https://jobs.company.com/role"
                className="w-full rounded-lg bg-neutral-900 border border-neutral-800 p-3 text-sm text-neutral-100 placeholder-neutral-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                onClick={async () => {
                  if (!jdUrl.trim()) return;

                  try {
                    toast.loading("Fetching job description...");
                    const data = await fetchJDFromUrl(jdUrl);
                    setJobDescription(data.job_description);
                    setJdMode("paste");
                    toast.dismiss();
                    toast.success("Job description extracted");
                  } catch (err: any) {
                    toast.dismiss();
                    toast.error(err.message);
                  }
                }}
                className="mt-2 text-xs text-blue-400 hover:underline"
              >
                Extract Job Description
              </button>

            </>
          )}
        </div>

        {/* Resume */}
        <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-4">
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm font-medium text-neutral-300">
              Resume
            </label>

            <div className="flex gap-2 mb-3">
              <button
                onClick={() => setResumeMode("paste")}
                className={`px-3 py-1.5 rounded-md text-sm ${
                  resumeMode === "paste"
                    ? "bg-blue-600 text-white"
                    : "bg-neutral-800 text-neutral-400"
                }`}
              >
                Paste Resume
              </button>

              <button
                onClick={() => setResumeMode("upload")}
                className={`px-3 py-1.5 rounded-md text-sm ${
                  resumeMode === "upload"
                    ? "bg-blue-600 text-white"
                    : "bg-neutral-800 text-neutral-400"
                }`}
              >
                Upload File
              </button>
            </div>

          </div>

          {resumeMode === "paste" && (
            <textarea
              value={resumeText}
              onChange={(e) => setResumeText(e.target.value)}
              rows={8}
              placeholder="Paste your resume content here…"
              className="w-full resize-none rounded-lg bg-neutral-900 border border-neutral-800 p-3 text-sm text-neutral-100 placeholder-neutral-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          )}

          {resumeMode === "upload" && (
            <input
              type="file"
              accept=".pdf,.doc,.docx,.txt,.md"
              disabled={uploading}
              onChange={async (e) => {
                const file = e.target.files?.[0];
                if (!file) return;

                try {
                  setUploading(true);
                  toast.loading("Extracting resume…");

                  const data = await uploadResumeFile(file);
                  setResumeText(data.text);
                  setResumeMode("paste");

                  toast.dismiss();
                  toast.success("Resume extracted successfully");
                } catch (err: any) {
                  toast.dismiss();
                  toast.error(err.message || "Failed to upload resume");
                } finally {
                  setUploading(false);
                }
              }}
            />
          )}



        </div>


          {/* Left – Action */}
          <div className="lg:col-span-1 rounded-2xl border border-neutral-800 bg-neutral-900 p-6 space-y-4">
            <h2 className="text-lg font-medium">Start Optimization</h2>
            <p className="text-sm text-neutral-400">
              Uses your resume + job description to generate optimized output.
            </p>

            <button
              onClick={startJob}
              disabled={
                loading ||
                uploading ||
                !resumeText.trim() ||
                (jdMode === "paste" && !jobDescription.trim()) ||
                (jdMode === "url" && !jdUrl.trim())
              }
              
              className="w-full rounded-lg bg-blue-600 px-6 py-3 text-sm font-medium hover:bg-blue-500 disabled:opacity-50"
            >
              {loading ? "Processing…" : "Start Optimization"}
            </button>
          </div>

          {/* Right Top – Status */}
          {jobStatus && (
            <div className="lg:col-span-2 rounded-2xl border border-neutral-800 bg-neutral-900 p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-medium">Job Status</h2>
                <span
                  className={`rounded-full px-3 py-1 text-xs font-medium ${
                    jobStatus.status === "success"
                      ? "bg-green-500/10 text-green-400"
                      : jobStatus.status === "failed"
                      ? "bg-red-500/10 text-red-400"
                      : "bg-blue-500/10 text-blue-400"
                  }`}
                >
                  {jobStatus.status}
                </span>
              </div>

              <div className="h-2 w-full overflow-hidden rounded bg-neutral-800">
                <div
                  className={`h-full transition-all duration-500 ${
                    jobStatus.status === "failed"
                      ? "bg-red-500"
                      : jobStatus.status === "success"
                      ? "bg-green-500"
                      : "bg-blue-500 animate-pulse"
                  }`}
                  style={{ width: `${progress}%` }}
                />
              </div>

              <p className="text-xs text-neutral-400">
                {jobStatus.status === "pending" && "Queued…"}
                {jobStatus.status === "running" && "Optimizing resume…"}
                {jobStatus.status === "success" && "Completed successfully"}
                {jobStatus.status === "failed" && "Job failed"}
              </p>
            </div>
          )}

          {/* Results */}
          {jobStatus?.status === "success" && jobStatus.result && (
            <div className="lg:col-span-3 rounded-2xl border border-neutral-800 bg-neutral-900 p-6 space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold">Results</h2>
                <button
                  onClick={() => {
                    setJobId(null);
                    setJobStatus(null);
                    setActiveTab("resume");
                  }}
                  className="text-sm text-blue-400 hover:underline"
                >
                  Start new
                </button>
              </div>

              {/* Tabs */}
              <div className="flex gap-6 border-b border-neutral-800">
                <button
                  onClick={() => setActiveTab("resume")}
                  className={`pb-2 text-sm ${
                    activeTab === "resume"
                      ? "border-b-2 border-blue-500 font-medium"
                      : "text-neutral-500"
                  }`}
                >
                  Resume
                </button>

                {jobStatus.result.cover_letter && (
                  <button
                    onClick={() => setActiveTab("cover")}
                    className={`pb-2 text-sm ${
                      activeTab === "cover"
                        ? "border-b-2 border-blue-500 font-medium"
                        : "text-neutral-500"
                    }`}
                  >
                    Cover Letter
                  </button>
                )}
              </div>

              {/* ATS Score */}
              {activeTab === "resume" &&
                jobStatus.result.old_ats_score !== undefined &&
                jobStatus.result.new_ats_score !== undefined && (
                  <p className="text-sm text-neutral-400">
                    ATS Score{" "}
                    <span className="line-through mx-2">
                      {jobStatus.result.old_ats_score}%
                    </span>
                    <span className="font-semibold text-green-400">
                      {jobStatus.result.new_ats_score}%
                    </span>
                  </p>
                )}

              {/* Content */}
              {activeTab === "resume" && jobStatus.result.optimized_resume && (
                <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-5 space-y-4 prose prose-invert max-w-none">
                  <ReactMarkdown>
                    {jobStatus.result.optimized_resume}
                  </ReactMarkdown>
                  <div className="flex gap-4 text-sm">
                    <button
                      onClick={() =>
                        copyToClipboard(jobStatus.result!.optimized_resume!)
                      }
                      className="text-blue-400 hover:underline"
                    >
                      Copy
                    </button>
                    <button
                      onClick={() =>
                        download(
                          jobStatus.result!.optimized_resume!,
                          "optimized_resume.md"
                        )
                      }
                      className="text-blue-400 hover:underline"
                    >
                      Download
                    </button>
                  </div>
                </div>
              )}

              {activeTab === "cover" && jobStatus.result.cover_letter && (
                <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-5 space-y-4 prose prose-invert max-w-none">
                  <ReactMarkdown>
                    {jobStatus.result.cover_letter}
                  </ReactMarkdown>
                  <div className="flex gap-4 text-sm">
                    <button
                      onClick={() =>
                        copyToClipboard(jobStatus.result!.cover_letter!)
                      }
                      className="text-blue-400 hover:underline"
                    >
                      Copy
                    </button>
                    <button
                      onClick={() =>
                        download(
                          jobStatus.result!.cover_letter!,
                          "cover_letter.md"
                        )
                      }
                      className="text-blue-400 hover:underline"
                    >
                      Download
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {error && (
          <div className="mt-6 rounded-lg border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-400">
            {error}
          </div>
        )}
      </div>
    </main>
  );

}
