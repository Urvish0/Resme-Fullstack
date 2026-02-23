"""
Structured JSON schema for resume data.

This is the "headless" representation of a resume — agent nodes produce
this structure, and rendering backends (Markdown, PDF, LaTeX) consume it.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
    name: str = Field(..., description="Full name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="City, State or City, Country")
    linkedin: Optional[str] = Field(None, description="LinkedIn URL or handle")
    github: Optional[str] = Field(None, description="GitHub URL or handle")
    portfolio: Optional[str] = Field(None, description="Portfolio/website URL")


class SkillCategory(BaseModel):
    category: str = Field(..., description="e.g. Languages, Frameworks, Tools")
    skills: List[str] = Field(default_factory=list)


class ExperienceItem(BaseModel):
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    location: Optional[str] = Field(None)
    start_date: str = Field(..., description="Start date, e.g. Jan 2023")
    end_date: str = Field("Present", description="End date or 'Present'")
    bullets: List[str] = Field(default_factory=list, description="Achievement bullets")


class EducationItem(BaseModel):
    degree: str = Field(..., description="Degree name, e.g. B.S. Computer Science")
    institution: str = Field(..., description="University/college name")
    location: Optional[str] = Field(None)
    year: Optional[str] = Field(None, description="Graduation year or date range")
    gpa: Optional[str] = Field(None)
    details: List[str] = Field(default_factory=list, description="Relevant coursework, honors, etc.")


class ProjectItem(BaseModel):
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="One-line description")
    technologies: List[str] = Field(default_factory=list)
    bullets: List[str] = Field(default_factory=list)
    url: Optional[str] = Field(None)


class CertificationItem(BaseModel):
    name: str = Field(..., description="Certification name")
    issuer: Optional[str] = Field(None, description="Issuing organization")
    date: Optional[str] = Field(None)


class ResumeJSON(BaseModel):
    """
    The canonical structured representation of a resume.
    All rendering backends (Markdown, PDF, HTML) consume this schema.
    """
    contact: ContactInfo
    summary: Optional[str] = Field(None, description="Professional summary paragraph")
    skills: List[SkillCategory] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    projects: List[ProjectItem] = Field(default_factory=list)
    certifications: List[CertificationItem] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the structured resume as Markdown (backward-compatible)."""
        lines: List[str] = []

        # Header
        c = self.contact
        contact_parts = [p for p in [c.location, c.email, c.phone, c.linkedin, c.github, c.portfolio] if p]
        lines.append(f"# {c.name}")
        if contact_parts:
            lines.append(" | ".join(contact_parts))
        lines.append("")
        lines.append("---")
        lines.append("")

        # Summary
        if self.summary:
            lines.append("## Professional Summary")
            lines.append("")
            lines.append(self.summary)
            lines.append("")

        # Skills
        if self.skills:
            lines.append("## Technical Skills")
            lines.append("")
            for cat in self.skills:
                lines.append(f"- **{cat.category}:** {', '.join(cat.skills)}")
            lines.append("")

        # Experience
        if self.experience:
            lines.append("## Professional Experience")
            lines.append("")
            for exp in self.experience:
                title_line = f"### {exp.title} | {exp.company}"
                if exp.location:
                    title_line += f" | {exp.location}"
                lines.append(title_line)
                lines.append(f"*{exp.start_date} - {exp.end_date}*")
                lines.append("")
                for bullet in exp.bullets:
                    lines.append(f"- {bullet}")
                lines.append("")

        # Education
        if self.education:
            lines.append("## Education")
            lines.append("")
            for edu in self.education:
                edu_line = f"### {edu.degree} | {edu.institution}"
                if edu.location:
                    edu_line += f" | {edu.location}"
                lines.append(edu_line)
                if edu.year:
                    lines.append(f"*{edu.year}*")
                if edu.gpa:
                    lines.append(f"GPA: {edu.gpa}")
                for detail in edu.details:
                    lines.append(f"- {detail}")
                lines.append("")

        # Projects
        if self.projects:
            lines.append("## Projects")
            lines.append("")
            for proj in self.projects:
                lines.append(f"### {proj.name}")
                if proj.description:
                    lines.append(proj.description)
                if proj.technologies:
                    lines.append(f"**Technologies:** {', '.join(proj.technologies)}")
                for bullet in proj.bullets:
                    lines.append(f"- {bullet}")
                if proj.url:
                    lines.append(f"[Link]({proj.url})")
                lines.append("")

        # Certifications
        if self.certifications:
            lines.append("## Certifications")
            lines.append("")
            for cert in self.certifications:
                parts = [cert.name]
                if cert.issuer:
                    parts.append(cert.issuer)
                if cert.date:
                    parts.append(cert.date)
                lines.append(f"- {', '.join(parts)}")
            lines.append("")

        return "\n".join(lines)
