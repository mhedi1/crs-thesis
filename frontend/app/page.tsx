import { Fragment } from "react";
import Link from "next/link";

// The landing page commits to the project's dark navy regardless of the
// viewer's theme, so colours are set explicitly rather than through the
// light/dark tokens in globals.css.
const NAVY = "#0f172a";
const CARD = "#121b2e";
const CARD_EMPHASIS = "#14203a";
const EDGE = "#1e2a42";
const EDGE_STRONG = "#2b3a55";
const EDGE_EMPHASIS = "#2a4a7f";
const TEXT = "#f8fafc";
const MUTED = "#94a3b8";
const FAINT = "#7c8aa0";
const ACCENT = "#3b82f6";
const ACCENT_TEXT = "#60a5fa";

type WorkflowStep = {
  label: string;
  title: string;
  body: string;
  emphasis: boolean;
};

const WORKFLOW: WorkflowStep[] = [
  {
    label: "STAGE 1",
    title: "Candidate Retrieval and Fusion",
    body: "KBRD and conversation-conditioned CKG rankings are combined using Reciprocal Rank Fusion.",
    emphasis: false,
  },
  {
    label: "STAGE 2",
    title: "Contextual Reranking",
    body: "A LoRA-adapted Qwen2.5-3B model reranks candidates using the dialogue context.",
    emphasis: true,
  },
  {
    label: "OUTPUT",
    title: "Conversational Response",
    body: "The selected movie is presented through a natural-language response, with support for follow-up interaction.",
    emphasis: false,
  },
];

export default function LandingPage() {
  return (
    <div
      className="home-scale flex min-h-screen w-full flex-col items-center justify-center px-5 py-[var(--home-pad)] sm:px-8"
      style={{ backgroundColor: NAVY, color: TEXT }}
    >
      <div className="flex w-full max-w-[1400px] flex-col items-center">
        <main className="flex w-full flex-col items-center text-center">
          <span
            className="inline-flex items-center gap-2.5 rounded-full border px-5 py-1.5 font-mono text-[0.72rem] uppercase tracking-[0.18em]"
            style={{ borderColor: EDGE_STRONG, color: MUTED }}
          >
            <span
              aria-hidden="true"
              className="block size-1.5 rounded-full"
              style={{ backgroundColor: ACCENT }}
            />
            Research Prototype
          </span>

          <h1
            className="mt-[var(--home-gap-badge)] max-w-[13ch] text-balance font-bold leading-[1.04] tracking-[-0.025em] lg:max-w-[680px] xl:max-w-[830px]"
            style={{ color: TEXT, fontSize: "var(--home-h1)" }}
          >
            Discover films through conversation.
          </h1>

          <p
            className="mt-[var(--home-gap-head)] max-w-[54ch] text-pretty leading-[1.7] xl:max-w-[860px]"
            style={{ color: MUTED, fontSize: "var(--home-sub)" }}
          >
            A two-stage conversational movie recommender combining structured
            retrieval with contextual LLM reranking.
          </p>

          <Link
            href="/chat"
            className="mt-[var(--home-gap-sub)] inline-flex items-center gap-3 rounded-full px-12 py-[var(--home-cta-pad)] text-base font-semibold text-white transition-colors hover:bg-[#2563eb] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#60a5fa] focus-visible:ring-offset-4 focus-visible:ring-offset-[#0f172a]"
            style={{ backgroundColor: ACCENT }}
          >
            Start Conversation
            <span aria-hidden="true" className="text-[1.05rem] leading-none">
              &rarr;
            </span>
          </Link>
        </main>

        <section className="mt-[var(--home-gap-section)] flex w-full flex-col items-center">
          <div className="flex w-full items-center gap-6">
            <div className="h-px flex-1" style={{ backgroundColor: EDGE }} />
            <h2
              className="shrink-0 font-mono text-[0.72rem] uppercase tracking-[0.22em]"
              style={{ color: FAINT }}
            >
              System Workflow
            </h2>
            <div className="h-px flex-1" style={{ backgroundColor: EDGE }} />
          </div>

          <div className="mt-[var(--home-gap-cards)] flex w-full flex-col items-stretch lg:flex-row">
            {WORKFLOW.map((step, index) => (
              <Fragment key={step.label}>
                {index > 0 && (
                  <div
                    aria-hidden="true"
                    className="flex shrink-0 items-center justify-center py-3 lg:w-16 lg:py-0"
                  >
                    <div
                      className="h-6 w-px lg:h-px lg:w-auto lg:flex-1"
                      style={{ backgroundColor: EDGE_STRONG }}
                    />
                    <span
                      className="hidden size-[7px] rotate-45 border-r border-t lg:block"
                      style={{ borderColor: EDGE_STRONG }}
                    />
                  </div>
                )}
                <article
                  className="flex flex-1 flex-col rounded-lg border p-[var(--home-card-pad)] text-left"
                  style={{
                    backgroundColor: step.emphasis ? CARD_EMPHASIS : CARD,
                    borderColor: step.emphasis ? EDGE_EMPHASIS : EDGE,
                    borderTopWidth: step.emphasis ? "2px" : undefined,
                    borderTopColor: step.emphasis ? ACCENT : undefined,
                  }}
                >
                  <span
                    className="font-mono text-[0.72rem] uppercase tracking-[0.18em]"
                    style={{ color: step.emphasis ? ACCENT_TEXT : MUTED }}
                  >
                    {step.label}
                  </span>
                  <h3
                    className="mt-4 text-pretty font-bold leading-snug"
                    style={{ color: TEXT, fontSize: "var(--home-card-title)" }}
                  >
                    {step.title}
                  </h3>
                  <p
                    className="mt-3 text-pretty leading-[1.65]"
                    style={{ color: MUTED, fontSize: "var(--home-card-body)" }}
                  >
                    {step.body}
                  </p>
                </article>
              </Fragment>
            ))}
          </div>
        </section>

        <footer className="mt-[var(--home-gap-footer)] w-full">
          <div className="h-px w-full" style={{ backgroundColor: EDGE }} />
          <p
            className="mt-[var(--home-gap-footer-text)] flex flex-wrap items-center justify-center gap-x-4 gap-y-1 text-center text-[0.9rem]"
            style={{ color: MUTED }}
          >
            <span>Pristini School of AI</span>
            <span aria-hidden="true" style={{ color: EDGE_STRONG }}>
              &middot;
            </span>
            <span>Universidad de Ja&eacute;n</span>
            <span aria-hidden="true" style={{ color: EDGE_STRONG }}>
              &middot;
            </span>
            <span className="font-mono tracking-[0.02em]">
              Mohamed Hedi Foughali
            </span>
          </p>
        </footer>
      </div>
    </div>
  );
}
