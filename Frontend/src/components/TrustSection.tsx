import { motion } from "framer-motion";
import { Shield, Eye, Lock, FileCode, UserCheck } from "lucide-react";

const guarantees = [
  {
    icon: <Shield className="w-5 h-5" />,
    title: "No training on your code",
    description: "Your IP remains yours. Zero data retention for model training."
  },
  {
    icon: <Lock className="w-5 h-5" />,
    title: "Optional local-only execution",
    description: "Run Sentinel strictly within your perimeter."
  },
  {
    icon: <Eye className="w-5 h-5" />,
    title: "Read-only repository access",
    description: "Sentinel cannot commit without approval."
  },
  {
    icon: <FileCode className="w-5 h-5" />,
    title: "Full diff transparency",
    description: "Review every character of every proposed fix."
  },
  {
    icon: <UserCheck className="w-5 h-5" />,
    title: "Human approval for every fix",
    description: "You remain the final gatekeeper."
  }
];

const TrustSection = () => {
  return (
    <section id="trust" className="section-padding relative overflow-hidden bg-card/30">
      <div className="container mx-auto px-6 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center max-w-3xl mx-auto mb-16"
        >
          <h2 className="font-display text-3xl sm:text-4xl lg:text-display-md font-bold leading-tight mb-4 text-foreground">
            Designed for Trust
          </h2>
          <p className="text-xl text-muted-foreground">
            Security tools require absolute transparency.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-5xl mx-auto">
          {guarantees.map((item, index) => (
            <motion.div
              key={item.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="p-6 rounded-xl border border-line-muted bg-background hover:border-primary/30 transition-colors"
            >
              <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary mb-4">
                {item.icon}
              </div>
              <h3 className="font-display font-semibold text-lg mb-2">{item.title}</h3>
              <p className="text-sm text-muted-foreground">{item.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default TrustSection;