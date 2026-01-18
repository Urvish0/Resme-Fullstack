"use client";

import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown"
import {
  optimizeResumeAsync,
  getJobStatus,
} from "@/lib/api";
import type { JobStatusResponse } from "@/lib/api";
export default function Home() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatusResponse | null>(null);
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<"resume" | "cover">("resume")
  const [progress, setProgress] = useState(0)

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
      setJobId(savedJobId)
    }
  }, [])
  
  const copyToClipboard = async (text: string) => {
    await navigator.clipboard.writeText(text)
    alert("Copied to clipboard")
  }

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
    //setResult(null)
    setProgress(0)
    setJobId(null)
    setActiveTab("resume")


    try {
      const res = await optimizeResumeAsync(
        {
          job_description: "We are looking for a backend engineer with strong experience in Python, FastAPI, Redis, AI systems, async programming, and production-grade APIs. The candidate should understand scalability, caching, and system design.",

          resume_text: "I am a backend-focused software engineer with experience in Python, FastAPI, Redis, async APIs, and AI-powered systems. I have built production services with caching and streaming.",

          resume_format: "markdown"
        },
        crypto.randomUUID()
      )

      setJobId(res.job_id)
      localStorage.setItem("jobId", res.job_id)
    } catch (e) {
      setError("Failed to start job")
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
          setError("Failed to fetch job status")
        }
        setLoading(false)
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [jobId])

  useEffect(() => {
    if (jobStatus?.status === "success") {
      window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" })
    }
  }, [jobStatus?.status])
  

  return (
    <main className="p-6 space-y-4">
      {!jobId && !loading && (
        <div className="rounded border border-dashed p-6 text-center text-gray-500">
          <p className="mb-2">No optimization running</p>
          <p className="text-sm">
            Click <strong>Start Optimization</strong> to generate your resume and cover letter.
          </p>
        </div>
      )}

      <button 
      onClick={startJob} 
      disabled={loading || ["pending", "running"].includes(jobStatus?.status ?? "")}>
        {loading ? "Processing..." : "Start Optimization"}
      </button>

      
      {jobStatus && (
        <div className="w-full max-w-md">
          <div className="h-2 w-full bg-neutral-700 rounded overflow-hidden">
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

          <p className="mt-1 text-xs text-gray-400">
            {jobStatus.status === "pending" && "Queued…"}
            {jobStatus.status === "running" && "Optimizing resume…"}
            {jobStatus.status === "success" && "Completed"}
            {jobStatus.status === "failed" && "Failed"}
          </p>
        </div>
      )}
      {jobStatus?.status === "pending" && (
        <p className="text-sm text-yellow-500">
          Job queued… starting shortly.
        </p>
      )}

      
      {jobId && <p>Job ID: {jobId}</p>}

      {jobStatus && (
        <>
          <p className="font-medium">
            Status:{" "}
            <span className={
              jobStatus.status === "success"
                ? "text-green-600"
                : jobStatus.status === "failed"
                ? "text-red-600"
                : "text-yellow-600"
            }>
              {jobStatus.status}
            </span>
          </p>

          {jobStatus?.status === "success" && !jobStatus.result && (
            <p className="text-yellow-500">
              Job completed, but no result returned.
            </p>
          )}

          {jobStatus.status === "success" && jobStatus.result && (
            <section className="space-y-4 border-t pt-4">
              <div className="flex gap-4 border-b">
                <button
                  onClick={() => setActiveTab("resume")}
                  disabled={jobStatus?.status !== "success"}
                  className={`pb-2 ${
                    activeTab === "resume"
                      ? "border-b-2 border-blue-500 font-semibold"
                      : "text-gray-500"
                  } disabled:opacity-50`}
                >
                  Resume
                </button>

                {jobStatus.result.cover_letter && (
                  <button
                    onClick={() => setActiveTab("cover")}
                    disabled={jobStatus?.status !== "success"}
                    className={`pb-2 ${
                      activeTab === "cover"
                        ? "border-b-2 border-blue-500 font-semibold"
                        : "text-gray-500"
                    } disabled:opacity-50`}
                  >
                    Cover Letter
                  </button>
                )}
              </div>


              <h2 className="text-lg font-semibold">
                {activeTab === "resume" ? "Optimized Resume" : "Cover Letter"}
              </h2>

              {jobStatus.result?.old_ats_score !== undefined &&
                jobStatus.result?.new_ats_score !== undefined && (
                  <p className="text-sm">
                    ATS Score:{" "}
                    <span className="line-through mr-2 text-gray-500">
                      {jobStatus.result.old_ats_score}%
                    </span>
                    <span className="text-green-500 font-semibold">
                      {jobStatus.result.new_ats_score}%
                    </span>
                  </p>
                )}

                {activeTab === "resume" && (
                  <>
                    {jobStatus.result.optimized_resume ? (
                      <div className="relative prose prose-invert max-w-none rounded bg-neutral-900 p-4 border border-neutral-700">
                        <ReactMarkdown>
                          {jobStatus.result.optimized_resume}
                        </ReactMarkdown>

                        {jobStatus.result.optimized_resume && (
                          <div className="mt-3 flex gap-4 text-sm">
                            <button
                            onClick={() =>
                              copyToClipboard(jobStatus.result!.optimized_resume!)
                            }
                            className="mt-2 text-sm text-blue-400 hover:underline"
                          >
                            Copy Resume
                          </button>
                          <button
                            onClick={() =>
                              download(jobStatus.result!.optimized_resume!, "optimized_resume.md")
                            }
                            className="mt-2 text-sm text-blue-400 hover:underline"
                          >
                            Download Resume
                          </button>
                          </div>
                        )}
                      </div>
                    ) : (
                      <p className="text-gray-500">No optimized resume returned.</p>
                    )}
                  </>
                )}

                {activeTab === "cover" && jobStatus.result.cover_letter && (
                  <div className="relative prose prose-invert max-w-none rounded bg-neutral-900 p-4 border border-neutral-700">
                    <ReactMarkdown>
                      {jobStatus.result.cover_letter}
                    </ReactMarkdown>
                    {jobStatus.result.cover_letter && (
                      <div className="mt-3 flex gap-4 text-sm">
                        <button
                          onClick={() =>
                            copyToClipboard(jobStatus.result!.cover_letter!)
                          }
                          className="mt-2 text-sm text-blue-400 hover:underline"
                        >
                          Copy Cover Letter
                        </button>
                        <button
                          onClick={() =>
                            download(jobStatus.result!.cover_letter!, "cover_letter.md")
                          }
                          className="mt-2 text-sm text-blue-400 hover:underline"
                        >
                          Download Cover Letter
                        </button>

                      </div>

                      )}
                  </div>
                )}


            </section>
          )}

          {jobStatus?.status === "success" && (
            <button
              onClick={() => {
                setJobId(null)
                setJobStatus(null)
                setActiveTab("resume")
                setError(null)
              }}
              className="mt-4 text-sm text-blue-400 hover:underline"
            >
              Start new optimization
            </button>
          )}



          {jobStatus.error && <p className="text-red-500">{jobStatus.error}</p>}
        </>
      )}

      {error && (
        <div className="rounded border border-red-400 bg-red-50 p-3 text-sm text-red-700">
          {error}
        </div>
      )}

    </main>
  )
}
