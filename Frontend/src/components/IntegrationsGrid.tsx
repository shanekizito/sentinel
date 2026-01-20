import { motion } from "framer-motion";
import { AWS, GitHub, Slack, Jira, VSCode } from "./BrandIcons";

const integrations = [
    { name: "GitHub", icon: GitHub },
    { name: "GitLab", icon: "https://cdn.simpleicons.org/gitlab/white" },
    { name: "Bitbucket", icon: "https://cdn.simpleicons.org/bitbucket/white" },
    { name: "VS Code", icon: VSCode },
    { name: "Linear", icon: "https://cdn.simpleicons.org/linear/white" },
    { name: "Jira", icon: Jira },
    { name: "Slack", icon: Slack },
    { name: "AWS", icon: AWS },
];

const IntegrationsGrid = () => {
    return (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {integrations.map((tool, idx) => {
                const isComponent = typeof tool.icon !== 'string';
                const IconComponent = isComponent ? tool.icon : null;

                return (
                    <motion.div
                        key={tool.name}
                        initial={{ opacity: 0, scale: 0.9 }}
                        whileInView={{ opacity: 1, scale: 1 }}
                        viewport={{ once: true }}
                        transition={{ delay: idx * 0.05 }}
                        className="group relative flex items-center gap-3 p-4 rounded-xl bg-card/30 border border-border/50 hover:bg-card/50 hover:border-primary/20 transition-all duration-300"
                    >
                        {/* Glow Effect */}
                        <div className="absolute inset-0 bg-gradient-to-r from-primary/0 via-primary/5 to-primary/0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-xl" />

                        <div className="relative w-8 h-8 flex items-center justify-center p-1.5 rounded-lg bg-black/50 border border-white/10 group-hover:scale-110 transition-transform duration-300">
                            {isComponent ? (
                                // @ts-ignore
                                <IconComponent className="w-full h-full text-white opacity-80 group-hover:opacity-100 transition-opacity" />
                            ) : (
                                <img
                                    src={tool.icon as string}
                                    alt={tool.name}
                                    className="w-full h-full object-contain opacity-80 group-hover:opacity-100 transition-opacity"
                                />
                            )}
                        </div>
                        <span className="relative text-sm font-medium text-muted-foreground group-hover:text-foreground transition-colors">
                            {tool.name}
                        </span>
                    </motion.div>
                );
            })}
        </div>
    );
};

export default IntegrationsGrid;
