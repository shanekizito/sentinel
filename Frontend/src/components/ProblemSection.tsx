import { AlertTriangle, Repeat, Clock, FileWarning, ArrowRight } from "lucide-react";

const ProblemSection = () => {
  return (
    <section className="py-32 bg-[#FDFDFB] border-y border-border/50 relative overflow-hidden">
      {/* Subtle Grid Background */}
      <div className="absolute inset-0 bg-grid-black/[0.02] -z-10" />

      <div className="container mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-16 lg:gap-24 items-start">

          {/* Left: The Narrative Impact Visual */}
          <div className="relative sticky top-32">
            <div className="mb-8">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-red-50 border border-red-100 text-xs font-bold text-red-600 mb-6 uppercase tracking-wide">
                <AlertTriangle className="w-3.5 h-3.5" />
                The Velocity Gap
              </div>
              <h2 className="font-display text-5xl font-bold mb-6 text-foreground tracking-tight leading-[1.1]">
                AI Writes Code Fast. <br />
                <span className="text-muted-foreground">Security Can't Keep Up.</span>
              </h2>
              <p className="text-xl text-muted-foreground leading-relaxed text-balance">
                AI has fundamentally changed the speed of development. Security workflows designed for human output are now the bottleneck.
              </p>
            </div>

            {/* Abstract Visual: Chaos vs Order */}
            <div className="relative h-64 w-full bg-gradient-to-br from-gray-900 to-black rounded-2xl overflow-hidden shadow-2xl border border-gray-800 p-8 flex flex-col justify-between group">
              <div className="text-gray-500 font-mono text-xs mb-2">/var/log/velocity.log</div>
              <div className="space-y-2 font-mono text-sm opacity-80">
                <div className="text-green-400">{">"} Generating feature... [Done 200ms]</div>
                <div className="text-green-400">{">"} Generating tests... [Done 50ms]</div>
                <div className="text-red-400 animate-pulse">{">"} WARN: Security scan pending...</div>
                <div className="text-red-500">{">"} ERR: Queue overflow (4,021 items)</div>
              </div>
              <div className="absolute right-0 bottom-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                <AlertTriangle className="w-32 h-32 text-white" />
              </div>
            </div>
          </div>

          {/* Right: The List (De-cardified) */}
          <div className="space-y-12 relative pt-8">
            {/* Connecting Line */}
            <div className="absolute left-[27px] top-10 bottom-10 w-0.5 bg-gray-200 hidden md:block" />

            {[
              {
                icon: <AlertTriangle className="w-6 h-6 text-red-600" />,
                title: "AI Optimizes for Output, Not Safety",
                desc: "LLMs prioritize functional correctness. Threat modeling, trust boundaries, and abuse cases are never part of the prompt by default.",
                quote: "Absent by design"
              },
              {
                icon: <Repeat className="w-6 h-6 text-orange-600" />,
                title: "Regeneration Re-introduces Risk",
                desc: "Even when fixed, regenerating a file can silently bring vulnerabilities back. Traditional tools assume code only moves forward; AI rewrites history.",
                quote: "Zombie vulnerabilities"
              },
              {
                icon: <Clock className="w-6 h-6 text-yellow-600" />,
                title: "Manual Review is a Bottleneck",
                desc: "Scans must be triggered. Audits scheduled. Fixes reviewed. When velocity increases 10x, anything manual gets skipped.",
                quote: "Security theater"
              },
              {
                icon: <FileWarning className="w-6 h-6 text-gray-600" />,
                title: "Tests Are Often Missing",
                desc: "Without regression tests, security fixes don't persist. They decay over time unnoticed. Security without tests is temporary.",
                quote: "No permanence"
              }
            ].map((item, index) => (
              <div key={index} className="relative flex gap-8 group">
                <div className="hidden md:flex w-14 h-14 rounded-full bg-white border-4 border-gray-50 items-center justify-center shrink-0 z-10 shadow-sm group-hover:scale-110 transition-transform duration-300">
                  {item.icon}
                </div>

                {/* Mobile Icon */}
                <div className="md:hidden mb-4">
                  {item.icon}
                </div>

                <div className="flex-1 pb-8 border-b border-gray-100 last:border-0 last:pb-0">
                  <h3 className="font-bold text-xl mb-3 group-hover:text-blue-600 transition-colors">{item.title}</h3>
                  <p className="text-muted-foreground leading-relaxed mb-3">
                    {item.desc}
                  </p>
                  <div className="flex items-center gap-2 text-xs font-bold text-foreground/70 uppercase tracking-wider">
                    <div className="w-1 h-1 rounded-full bg-red-500" />
                    {item.quote}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default ProblemSection;