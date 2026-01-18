import { motion } from "framer-motion";
import { Quote } from "lucide-react";

const testimonials = [
  {
    quote: "I don't have time to be wrong—or to rerun scans.",
    author: "Solo Founder",
    role: "Indie SaaS",
  },
  {
    quote: "Sentinel runs on every PR, automatically.",
    author: "Engineering Lead",
    role: "AI Startup",
  },
  {
    quote: "Security regressions don't sneak back in anymore.",
    author: "Security Engineer",
    role: "Fintech",
  },
];

const notFor = [
  "Teams looking for prettier code",
  "Style-focused refactors", 
  "Checkbox compliance theater",
  "One-off audits that rot instantly"
];

const Testimonials = () => {
  return (
    <section id="use-cases" className="py-24 relative overflow-hidden">
      <div className="absolute inset-0 bg-card/30" />
      
      <div className="container mx-auto px-6 relative z-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center max-w-2xl mx-auto mb-14"
        >
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-4">Built For</p>
          
          <h2 className="font-display text-3xl sm:text-4xl font-semibold mb-5 text-foreground tracking-tight">
            Who Uses Sentinel
          </h2>
          
          <p className="text-muted-foreground">
            Solo founders shipping MVPs. Startups using AI as teammates. 
            Teams that regenerate codebases weekly.
          </p>
        </motion.div>

        {/* Testimonials */}
        <div className="grid md:grid-cols-3 gap-5 max-w-4xl mx-auto mb-16">
          {testimonials.map((testimonial, index) => (
            <motion.div
              key={testimonial.author}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
            >
              <div className="h-full p-5 rounded-xl border border-white/5 bg-background/50">
                <Quote className="w-6 h-6 text-accent/30 mb-3" />
                
                <blockquote className="text-foreground font-medium mb-4 leading-relaxed">
                  "{testimonial.quote}"
                </blockquote>
                
                <div>
                  <p className="text-sm font-medium text-foreground">{testimonial.author}</p>
                  <p className="text-xs text-muted-foreground">{testimonial.role}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Not For */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center max-w-xl mx-auto"
        >
          <p className="text-xs font-medium text-muted-foreground/60 uppercase tracking-wider mb-3">
            Explicitly NOT for
          </p>
          <div className="flex flex-wrap items-center justify-center gap-2 mb-4">
            {notFor.map((item) => (
              <span key={item} className="px-3 py-1.5 rounded-full bg-muted/50 text-xs text-muted-foreground">
                {item}
              </span>
            ))}
          </div>
          <p className="text-sm text-foreground font-medium">
            Sentinel prevents real-world exploits, not impress linters.
          </p>
        </motion.div>
      </div>
    </section>
  );
};

export default Testimonials;
