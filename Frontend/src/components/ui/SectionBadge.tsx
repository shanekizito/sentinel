import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

interface SectionBadgeProps {
  icon?: LucideIcon;
  text: string;
  variant?: "cyber" | "premium" | "glass";
  className?: string;
}

const SectionBadge = ({
  icon: Icon,
  text,
  variant = "cyber",
  className,
}: SectionBadgeProps) => {
  const variants = {
    cyber: "badge-cyber text-primary",
    premium: "badge-premium text-violet",
    glass: "glass-card text-foreground",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5 }}
      className={cn(
        "inline-flex items-center gap-2 px-4 py-2 rounded-full",
        variants[variant],
        className
      )}
    >
      {Icon && <Icon className="w-4 h-4" />}
      <span className="text-xs font-semibold uppercase tracking-wider">{text}</span>
    </motion.div>
  );
};

export { SectionBadge };
