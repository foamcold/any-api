import { Link, useNavigate } from 'react-router-dom';
import { useEffect } from 'react';
import axios from 'axios';
import { useSystem } from '@/contexts/SystemContext';
import { API_BASE_URL } from '@/utils/api';
import { ArrowRight, Zap, Shield, Settings, Terminal, Users, Share2 } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function HomePage() {
    const isLoggedIn = !!localStorage.getItem('token');
    const navigate = useNavigate();
    const { config: systemConfig } = useSystem();
    const apiBaseUrl = systemConfig?.server_url || window.location.origin;

    // 检查系统是否需要初始化
    useEffect(() => {
        const checkSetupStatus = async () => {
            try {
                const response = await axios.get(`${API_BASE_URL}/setup/status`);
                if (response.data.needs_setup) {
                    navigate('/initialize');
                }
            } catch (error) {
                console.error('Failed to check setup status:', error);
            }
        };

        checkSetupStatus();
    }, [navigate]);

    return (
        <div className="h-screen bg-background flex flex-col overflow-hidden">
            {/* Navbar */}
            <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-50">
                <div className="px-4 h-16 flex items-center justify-between">
                    <a href="/" className="flex items-center gap-2 font-bold text-xl">
                        <img src="/vite.svg" alt="Logo" className="w-8 h-8" />
                        <span className="bg-clip-text text-transparent bg-gradient-to-r from-primary to-purple-600 dark:to-purple-400">
                            {systemConfig?.site_name || <>&nbsp;</>}
                        </span>
                    </a>
                    <div className="flex items-center gap-4">
                        <a href="https://github.com/foamcold/any-api" target="_blank" rel="noreferrer" className="text-sm font-medium hover:text-primary transition-colors">
                            GitHub
                        </a>
                        {isLoggedIn ? (
                            <Link to="/dashboard">
                                <Button>
                                    控制台
                                    <ArrowRight className="w-4 h-4 ml-2" />
                                </Button>
                            </Link>
                        ) : (
                            <Link to="/login">
                                <Button>登录</Button>
                            </Link>
                        )}
                    </div>
                </div>
            </header>

            {/* Hero Section */}
            <main className="flex-1 overflow-y-auto flex flex-col">
                <section className="py-20 lg:py-32 relative">
                    <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-background pointer-events-none" />
                    <div className="container mx-auto px-4 relative z-10 text-center">
                        <h1 className="text-4xl font-extrabold tracking-tight lg:text-6xl mb-6 bg-clip-text text-transparent bg-gradient-to-r from-primary to-purple-600 dark:to-purple-400">
                            Welcome to Any API
                        </h1>
                        <p className="text-xl text-muted-foreground mb-10 max-w-[800px] mx-auto leading-relaxed">
                            终极 Gemini API 代理解决方案，配备
                            <span className="text-foreground font-medium">预设管理</span>、
                            <span className="text-foreground font-medium">正则处理</span>、
                            <span className="text-foreground font-medium">渠道管理</span>等高级功能。
                            无缝兼容 OpenAI、Gemini 等多种格式。
                        </p>

                        <div className="flex flex-col sm:flex-row gap-4 justify-center mb-16">
                            <Link to={isLoggedIn ? "/dashboard" : "/login"}>
                                <Button size="lg" className="w-full sm:w-auto text-lg px-8 h-12">
                                    {isLoggedIn ? '前往控制台' : '免费开始使用'}
                                    <ArrowRight className="w-5 h-5 ml-2" />
                                </Button>
                            </Link>
                            <a href="#features">
                                <Button variant="outline" size="lg" className="w-full sm:w-auto text-lg px-8 h-12">
                                    了解更多
                                </Button>
                            </a>
                        </div>

                        {/* API Endpoint Card */}
                        <div className="bg-card border rounded-xl shadow-lg max-w-2xl mx-auto overflow-hidden text-left">
                            <div className="bg-muted px-4 py-3 border-b flex items-center gap-2">
                                <div className="w-3 h-3 rounded-full bg-red-500" />
                                <div className="w-3 h-3 rounded-full bg-yellow-500" />
                                <div className="w-3 h-3 rounded-full bg-green-500" />
                                <span className="ml-2 text-xs font-mono text-muted-foreground">api-endpoint</span>
                            </div>
                            <div className="p-6 font-mono text-sm space-y-4">
                                <div>
                                    <div className="text-muted-foreground mb-1">// OpenAI 兼容端点</div>
                                    <div className="bg-muted/50 p-3 rounded border flex items-center justify-between group">
                                        <code className="text-primary">{apiBaseUrl}/v1/chat/completions</code>
                                    </div>
                                </div>
                                <div>
                                    <div className="text-muted-foreground mb-1">// Gemini 兼容端点</div>
                                    <div className="bg-muted/50 p-3 rounded border flex items-center justify-between group">
                                        <code className="text-primary">{apiBaseUrl}/v1beta/models</code>
                                    </div>
                                </div>
                                <div>
                                    <div className="text-muted-foreground mb-1">// Claude 兼容端点</div>
                                    <div className="bg-muted/50 p-3 rounded border flex items-center justify-between group">
                                        <code className="text-primary">{apiBaseUrl}/v1/message</code>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Features Grid */}
                <section id="features" className="py-20 bg-muted/30">
                    <div className="container mx-auto px-4">
                        <h2 className="text-3xl font-bold text-center mb-12">功能亮点</h2>
                        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                            <FeatureCard
                                icon={<Zap className="w-10 h-10 text-orange-500" />}
                                title="多渠道互转"
                                description="无缝兼容 OpenAI, Gemini 等多种 API 格式，统一调用标准，轻松实现模型切换与聚合。"
                            />
                            <FeatureCard
                                icon={<Settings className="w-10 h-10 text-blue-500" />}
                                title="高级预设功能"
                                description="通过提示词注入、变量支持 ({{roll}}, {{random}}) 和上下文管理，动态构建和优化您的 API 请求。"
                            />
                            <FeatureCard
                                icon={<Terminal className="w-10 h-10 text-purple-500" />}
                                title="局部与全局正则"
                                description="在请求前后应用精细化的正则表达式，实现数据清洗、格式强制或内容审查，支持全局与密钥级配置。"
                            />
                            <FeatureCard
                                icon={<Shield className="w-10 h-10 text-green-500" />}
                                title="密钥与渠道管理"
                                description="精细化管理 API 密钥，密钥独立绑定渠道与预设，多次轮询失败自动禁用。"
                            />
                            <FeatureCard
                                icon={<Users className="w-10 h-10 text-teal-500" />}
                                title="多用户管理"
                                description="支持多用户和权限管理系统，为不同用户分配独立密钥和访问策略，实现资源隔离与安全控制。"
                            />
                            <FeatureCard
                                icon={<Share2 className="w-10 h-10 text-indigo-500" />}
                                title="日志与统计"
                                description="提供请求日志和数据统计，帮助您分析 API 使用情况、监控成本和诊断问题。"
                            />
                        </div>
                    </div>
                </section>
                {/* Footer */}
                <footer className="border-t py-8 bg-card mt-auto">
                    <div className="container mx-auto px-4 text-center text-muted-foreground text-sm">
                        <p>&copy; {new Date().getFullYear()} {systemConfig?.site_name || 'Any API'}。保留一切权利。</p>
                    </div>
                </footer>
            </main>
        </div>
    );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode, title: string, description: string }) {
    return (
        <div className="bg-card border rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="mb-4 bg-accent/50 w-16 h-16 rounded-full flex items-center justify-center">
                {icon}
            </div>
            <h3 className="text-xl font-semibold mb-2">{title}</h3>
            <p className="text-muted-foreground leading-relaxed">
                {description}
            </p>
        </div>
    );
}
