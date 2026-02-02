"use client";

import { useState, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import { motion, AnimatePresence, useScroll, useSpring } from "framer-motion";
import {
  FileText,
  Mail,
  Send,
  Upload,
  Link as LinkIcon,
  Clipboard,
  Download as DownloadIcon,
  RefreshCcw,
  Plus,
  ArrowRight,
} from "lucide-react";
import {
  optimizeResumeAsync,
  getJobStatus,
  uploadResumeFile,
  fetchJDFromUrl,
} from "@/lib/api";
import type { JobStatusResponse, OptimizeRequestExtended } from "@/lib/api";
import toast from "react-hot-toast";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export default function Home() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatusResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"resume" | "cover" | "cold">(
    "resume",
  );
  const [progress, setProgress] = useState(0);
  const [jobDescription, setJobDescription] = useState("");
  const [resumeText, setResumeText] = useState("");
  const [resumeMode, setResumeMode] = useState<"paste" | "upload">("paste");
  const [jdMode, setJdMode] = useState<"paste" | "url">("paste");
  const [jdUrl, setJdUrl] = useState("");
  const [uploading, setUploading] = useState(false);
  const [outputs, setOutputs] = useState({
    resume: true,
    cover: false,
    coldEmail: false,
  });
  const [coldSenderName, setColdSenderName] = useState("");
  const [coldSenderEmail, setColdSenderEmail] = useState("");
  const [coldRecipientName, setColdRecipientName] = useState("");
  const [coldRecipientEmail, setColdRecipientEmail] = useState("");
  const [coldCompanyName, setColdCompanyName] = useState("");
  const [coldTargetRole, setColdTargetRole] = useState("");

  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, {
    stiffness: 100,
    damping: 30,
    restDelta: 0.001,
  });

  const statusMessages = [
    "Analyzing semantic clusters...",
    "Aligning with ATS hierarchies...",
    "Synthesizing outreach narratives...",
    "Optimizing keyword density...",
    "Structuring professional narrative...",
    "Finalizing neural alignment...",
  ];

  const [statusMessageIndex, setStatusMessageIndex] = useState(0);

  useEffect(() => {
    if (!jobStatus) {
      setProgress(0);
      setStatusMessageIndex(0);
      return;
    }

    // Pseudo-progress for smoother animation
    let interval: NodeJS.Timeout;
    if (jobStatus.status === "pending" || jobStatus.status === "running") {
      interval = setInterval(() => {
        setProgress((prev) => {
          const cap = jobStatus.status === "pending" ? 30 : 90;
          if (prev < cap) return prev + Math.random() * 5; // Faster increment for better feedback
          return prev;
        });
        setStatusMessageIndex((prev) => (prev + 1) % statusMessages.length);
      }, 1500); // More frequent than the 2s polling to ensure ticks happen
    } else if (jobStatus.status === "success") {
      setProgress(100);
    }

    return () => clearInterval(interval);
  }, [jobStatus?.status]); // Critical: only reset when status actually changes

  useEffect(() => {
    const savedJobId = localStorage.getItem("jobId");
    if (savedJobId) {
      toast("Session restored.", { icon: "ℹ️" });
      localStorage.removeItem("jobId");
    }
  }, []);

  const copyToClipboard = async (text: string) => {
    await navigator.clipboard.writeText(text);
    toast.success("Copied");
  };

  const download = (content: string, filename: string) => {
    const blob = new Blob([content], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const startJob = async () => {
    setLoading(true);
    setError(null);
    setJobStatus(null);
    setProgress(0);
    setJobId(null);
    setActiveTab("resume");

    try {
      const services: string[] = [];
      if (outputs.resume) services.push("resume");
      if (outputs.cover) services.push("cover");
      if (outputs.coldEmail) services.push("coldEmail");

      const payload: OptimizeRequestExtended = {
        job_description: jobDescription,
        resume_text: resumeText,
        resume_format: "markdown",
        services,
        cold_email_sender_name: coldSenderName,
        cold_email_sender_email: coldSenderEmail,
        cold_email_recipient_name: coldRecipientName,
        cold_email_recipient_email: coldRecipientEmail,
        cold_email_company_name: coldCompanyName,
        cold_email_target_role: coldTargetRole,
      };

      const res = await optimizeResumeAsync(payload, crypto.randomUUID());
      setJobId(res.job_id);
      toast.success("Alignment Initiated");
      localStorage.setItem("jobId", res.job_id);
    } catch (e) {
      toast.error("Process Failed");
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!jobId) return;
    const interval = setInterval(async () => {
      try {
        const status = await getJobStatus(jobId);
        setJobStatus(status);
        if (status.status === "success" || status.status === "failed") {
          clearInterval(interval);
          setLoading(false);
          localStorage.removeItem("jobId");
        }
      } catch (e: any) {
        clearInterval(interval);
        setLoading(false);
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [jobId]);

  useEffect(() => {
    if (jobStatus?.status === "success") {
      toast.success("Alignment Complete", {
        duration: 5000,
        icon: "⚡",
        style: {
          border: "1px solid #fff",
          padding: "16px",
          color: "#fff",
          background: "#000",
        },
      });
      window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
      if (outputs.resume && jobStatus.result?.optimized_resume)
        setActiveTab("resume");
      else if (outputs.cover && jobStatus.result?.cover_letter)
        setActiveTab("cover");
      else if (outputs.coldEmail && jobStatus.result?.cold_email)
        setActiveTab("cold");
    }
  }, [jobStatus?.status]);

  const isFormValid =
    resumeText.trim() &&
    (jdMode === "paste" ? jobDescription.trim() : jdUrl.trim()) &&
    (!outputs.coldEmail || coldSenderName.trim());

  return (
    <main className="min-h-screen bg-black text-white selection:bg-white selection:text-black">
      {/* Progress Line */}
      <motion.div
        className="fixed top-0 left-0 right-0 h-[1px] bg-white z-50 origin-[0%]"
        style={{ scaleX }}
      />

      <div className="mx-auto max-w-[1600px] px-6 lg:px-12 py-12 lg:py-24">
        {/* Editorial Header */}
        <header className="grid grid-cols-1 lg:grid-cols-12 gap-12 mb-32 items-end">
          <div className="lg:col-span-8 space-y-8">
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-xs font-bold tracking-[0.3em] text-white/40 uppercase"
            >
              Precision Career Engineering — v2.0
            </motion.p>
            <h1 className="text-fluid-hero serif leading-[0.85] tracking-[-0.05em] -ml-2">
              ResMe <br />
              <span className="opacity-50 italic">Alignment</span>
            </h1>
          </div>
          <div className="lg:col-span-4 pb-4">
            <p className="text-fluid-base text-white/60 leading-relaxed font-medium">
              A minimalist framework for neural ATS optimization. Transforming
              scattered professional history into high-signal career assets.
            </p>
          </div>
        </header>

        {/* Unified Input Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-px bg-white/10 border border-white/10 mb-24 overflow-hidden">
          {/* Services Selection */}
          <section className="lg:col-span-3 bg-black p-8 lg:p-12 space-y-12">
            <h2 className="text-xs font-black tracking-widest text-white/30 uppercase">
              01 / Services
            </h2>
            <div className="space-y-2">
              <ServiceToggle
                label="Professional Resume"
                active={outputs.resume}
                onClick={() =>
                  setOutputs({ ...outputs, resume: !outputs.resume })
                }
              />
              <ServiceToggle
                label="Cover Letter"
                active={outputs.cover}
                onClick={() =>
                  setOutputs({ ...outputs, cover: !outputs.cover })
                }
              />
              <ServiceToggle
                label="Cold Outreach"
                active={outputs.coldEmail}
                onClick={() =>
                  setOutputs({ ...outputs, coldEmail: !outputs.coldEmail })
                }
              />
            </div>

            <AnimatePresence>
              {outputs.coldEmail && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className="space-y-4 pt-8"
                >
                  <MinimalInput
                    placeholder="Sender Name"
                    value={coldSenderName}
                    onChange={(e) => setColdSenderName(e.target.value)}
                  />
                  <MinimalInput
                    placeholder="Target Company"
                    value={coldCompanyName}
                    onChange={(e) => setColdCompanyName(e.target.value)}
                  />
                  <MinimalInput
                    placeholder="Target Role"
                    value={coldTargetRole}
                    onChange={(e) => setColdTargetRole(e.target.value)}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </section>

          {/* Job Specification */}
          <section className="lg:col-span-4 bg-black p-8 lg:p-12 space-y-12 border-x lg:border-white/10">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-black tracking-widest text-white/30 uppercase">
                02 / Specification
              </h2>
              <div className="flex gap-4">
                <button
                  onClick={() => setJdMode("paste")}
                  className={cn(
                    "text-[10px] font-bold uppercase tracking-widest",
                    jdMode === "paste" ? "text-white" : "text-white/20",
                  )}
                >
                  Paste
                </button>
                <button
                  onClick={() => setJdMode("url")}
                  className={cn(
                    "text-[10px] font-bold uppercase tracking-widest",
                    jdMode === "url" ? "text-white" : "text-white/20",
                  )}
                >
                  URL
                </button>
              </div>
            </div>

            <div className="h-[300px]">
              {jdMode === "paste" ? (
                <textarea
                  className="w-full h-full bg-transparent border-none resize-none focus:ring-0 text-fluid-base placeholder:text-white/10 p-0 font-medium"
                  placeholder="Insert job description here..."
                  value={jobDescription}
                  onChange={(e) => setJobDescription(e.target.value)}
                />
              ) : (
                <div className="space-y-6">
                  <MinimalInput
                    placeholder="https://job-posting.link"
                    value={jdUrl}
                    onChange={(e) => setJdUrl(e.target.value)}
                  />
                  <button
                    onClick={async () => {
                      if (!jdUrl.trim()) return;
                      const toastId = toast.loading("Fetching metadata");
                      try {
                        const data = await fetchJDFromUrl(jdUrl);
                        setJobDescription(data.job_description);
                        setJdMode("paste");
                        toast.success("Merged", { id: toastId });
                      } catch {
                        toast.error("Failed", { id: toastId });
                      }
                    }}
                    className="w-full py-4 border border-white/10 text-[10px] font-black uppercase tracking-[0.2em] hover:bg-white hover:text-black transition-all"
                  >
                    Capture Specification
                  </button>
                </div>
              )}
            </div>
          </section>

          {/* Resume History */}
          <section className="lg:col-span-5 bg-black p-8 lg:p-12 space-y-12">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-black tracking-widest text-white/30 uppercase">
                03 / History
              </h2>
              <div className="flex gap-4">
                <button
                  onClick={() => setResumeMode("paste")}
                  className={cn(
                    "text-[10px] font-bold uppercase tracking-widest",
                    resumeMode === "paste" ? "text-white" : "text-white/20",
                  )}
                >
                  Text
                </button>
                <button
                  onClick={() => setResumeMode("upload")}
                  className={cn(
                    "text-[10px] font-bold uppercase tracking-widest",
                    resumeMode === "upload" ? "text-white" : "text-white/20",
                  )}
                >
                  Document
                </button>
              </div>
            </div>

            <div className="h-[300px]">
              {resumeMode === "paste" ? (
                <textarea
                  className="w-full h-full bg-transparent border-none resize-none focus:ring-0 text-fluid-base placeholder:text-white/10 p-0 font-medium"
                  placeholder="Insert resume content here..."
                  value={resumeText}
                  onChange={(e) => setResumeText(e.target.value)}
                />
              ) : (
                <div className="h-full border border-dashed border-white/10 flex flex-col items-center justify-center p-8 group relative cursor-pointer hover:border-white/30 transition-all">
                  <Upload className="w-8 h-8 opacity-20 group-hover:opacity-100 transition-opacity mb-4" />
                  <p className="text-[10px] font-black uppercase tracking-widest opacity-40">
                    Drop PDF / DOCX
                  </p>
                  <input
                    type="file"
                    className="absolute inset-0 opacity-0 cursor-pointer"
                    disabled={uploading}
                    onChange={async (e) => {
                      const file = e.target.files?.[0];
                      if (!file) return;
                      setUploading(true);
                      const tId = toast.loading("Processing");
                      try {
                        const data = await uploadResumeFile(file);
                        setResumeText(data.text);
                        setResumeMode("paste");
                        toast.success("Processed", { id: tId });
                      } catch {
                        toast.error("Error", { id: tId });
                      } finally {
                        setUploading(false);
                      }
                    }}
                  />
                </div>
              )}
            </div>
          </section>
        </div>

        {/* Execution Control */}
        <div className="flex flex-col lg:flex-row items-center justify-between gap-12 mb-32 border-t border-white/10 pt-12">
          <div className="max-w-md">
            <h3 className="text-fluid-base font-bold mb-2 text-white">
              Execute Alignment
            </h3>
            <p className="text-[10px] text-white/40 uppercase tracking-widest font-black">
              Neural ATS Synchronicity
            </p>
          </div>
          <button
            onClick={startJob}
            disabled={!isFormValid || loading}
            className={cn(
              "px-12 py-6 text-fluid-xl serif italic font-medium transition-all flex items-center gap-6",
              isFormValid && !loading
                ? "bg-white text-black hover:pr-16"
                : "bg-white/5 text-white/20 cursor-not-allowed",
            )}
          >
            {loading ? "Processing..." : "Run Optimization"}
            <ArrowRight className="w-8 h-8" />
          </button>
        </div>

        {/* Real-time Status */}
        <AnimatePresence>
          {jobId && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-32 grid grid-cols-1 lg:grid-cols-12 gap-8 items-end"
            >
              <div className="lg:col-span-8 space-y-4">
                <div className="flex items-center gap-4">
                  <span className="w-2 h-2 rounded-full bg-white animate-pulse" />
                  <span className="text-xs font-black uppercase tracking-[0.2em]">
                    {statusMessages[statusMessageIndex]}
                  </span>
                </div>
                <div className="w-full h-px bg-white/10 relative">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    className="absolute top-0 left-0 h-full bg-white shadow-[0_0_20px_rgba(255,255,255,0.5)]"
                  />
                </div>
              </div>
              <div className="lg:col-span-4 text-right">
                <span className="text-fluid-3xl serif italic">
                  {Math.round(progress)}%
                </span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results Sections */}
        <AnimatePresence>
          {jobStatus?.status === "success" && jobStatus.result && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-24"
            >
              <div className="border-t border-white/10 pt-12 flex items-end justify-between">
                <h2 className="text-fluid-3xl serif tracking-tight">
                  Alignment <br />{" "}
                  <span className="opacity-50 italic">Complete</span>
                </h2>
                <button
                  onClick={() => {
                    setJobId(null);
                    setJobStatus(null);
                    setJobDescription("");
                    setResumeText("");
                    setJdUrl("");
                    setColdSenderName("");
                    setColdSenderEmail("");
                    setColdRecipientName("");
                    setColdRecipientEmail("");
                    setColdCompanyName("");
                    setColdTargetRole("");
                    setOutputs({
                      resume: true,
                      cover: false,
                      coldEmail: false,
                    });
                    setActiveTab("resume");
                    setProgress(0);
                    window.scrollTo({ top: 0, behavior: "smooth" });
                  }}
                  className="px-6 py-3 border border-white/10 text-[10px] font-black uppercase tracking-widest hover:bg-white hover:text-black transition-all"
                >
                  Reset Framework
                </button>
              </div>

              {/* Minimal Tabs */}
              <div className="flex gap-12 border-b border-white/10">
                {outputs.resume && jobStatus.result.optimized_resume && (
                  <TabItem
                    active={activeTab === "resume"}
                    onClick={() => setActiveTab("resume")}
                    label="Resume"
                    score={jobStatus.result.new_ats_score}
                  />
                )}
                {outputs.cover && jobStatus.result.cover_letter && (
                  <TabItem
                    active={activeTab === "cover"}
                    onClick={() => setActiveTab("cover")}
                    label="Letter"
                  />
                )}
                {outputs.coldEmail && jobStatus.result.cold_email && (
                  <TabItem
                    active={activeTab === "cold"}
                    onClick={() => setActiveTab("cold")}
                    label="Outreach"
                  />
                )}
              </div>

              {/* Dynamic Content */}
              <div className="max-w-4xl mx-auto py-12">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={activeTab}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    className="space-y-12"
                  >
                    <div className="flex items-center justify-between">
                      <p className="text-[10px] font-black uppercase tracking-[0.3em] opacity-30">
                        Aligned Output // {activeTab}
                      </p>
                      <div className="flex gap-4">
                        <IconButton
                          icon={<Clipboard className="w-4 h-4" />}
                          onClick={() =>
                            copyToClipboard(
                              activeTab === "resume"
                                ? jobStatus.result?.optimized_resume || ""
                                : activeTab === "cover"
                                  ? jobStatus.result?.cover_letter || ""
                                  : jobStatus.result?.cold_email || "",
                            )
                          }
                        />
                        <IconButton
                          icon={<DownloadIcon className="w-4 h-4" />}
                          onClick={() =>
                            download(
                              activeTab === "resume"
                                ? jobStatus.result?.optimized_resume || ""
                                : activeTab === "cover"
                                  ? jobStatus.result?.cover_letter || ""
                                  : jobStatus.result?.cold_email || "",
                              `${activeTab}.md`,
                            )
                          }
                        />
                      </div>
                    </div>
                    <div className="prose-minimal">
                      <ReactMarkdown>
                        {activeTab === "resume"
                          ? jobStatus.result.optimized_resume || ""
                          : activeTab === "cover"
                            ? jobStatus.result.cover_letter || ""
                            : jobStatus.result.cold_email || ""}
                      </ReactMarkdown>
                    </div>
                  </motion.div>
                </AnimatePresence>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </main>
  );
}

// Minimal Components
function ServiceToggle({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full py-4 px-6 border border-white/5 transition-all flex items-center justify-between group",
        active ? "bg-white text-black border-white" : "hover:border-white/20",
      )}
    >
      <span className="text-[11px] font-bold uppercase tracking-widest">
        {label}
      </span>
      {active ? (
        <Plus className="w-4 h-4 rotate-45" />
      ) : (
        <Plus className="w-4 h-4 opacity-20 group-hover:opacity-100" />
      )}
    </button>
  );
}

function MinimalInput({
  placeholder,
  value,
  onChange,
}: {
  placeholder: string;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <div className="relative border-b border-white/20 focus-within:border-white transition-all py-2">
      <input
        className="w-full bg-transparent border-none focus:ring-0 p-0 text-sm placeholder:text-white/40 text-white font-medium"
        placeholder={placeholder}
        value={value}
        onChange={onChange}
      />
    </div>
  );
}

function TabItem({
  label,
  active,
  onClick,
  score,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
  score?: number;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "pb-6 text-fluid-base font-bold transition-all relative",
        active ? "text-white" : "text-white/20 hover:text-white/40",
      )}
    >
      {label}
      {score && (
        <span className="absolute -top-4 -right-6 text-[10px] font-black text-white/30">
          {score}%
        </span>
      )}
      {active && (
        <motion.div
          layoutId="tab-underline"
          className="absolute bottom-0 left-0 right-0 h-1 bg-white"
        />
      )}
    </button>
  );
}

function IconButton({
  icon,
  onClick,
}: {
  icon: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="w-10 h-10 border border-white/10 flex items-center justify-center hover:bg-white hover:text-black transition-all"
    >
      {icon}
    </button>
  );
}
