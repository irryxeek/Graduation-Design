import sys

with open(r'D:\02_Study\01_Schoolwork\Graduation Design\docs\midterm\林逸飞-220110814-本科毕业设计中期报告.txt', 'r', encoding='utf-8') as f:
    text = f.read()

replacements = {
    'log₁₀(|BA| + 1×10⁻⁶)': r'$\log_{10}(|BA| + 1\times 10^{-6})$',
    'log₁₀(P)': r'$\log_{10}(P)$',
    "x' = (x - μ) / σ": r"$x' = \frac{x - \mu}{\sigma}$",
    'q(xₜ | xₜ₋₁) = N(xₜ; √(1-βₜ)·xₜ₋₁, βₜ·I)': r'$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})$',
    '{βₜ}_{t=1}^T': r'$\{\beta_t\}_{t=1}^T$',
    'β₁=1×10⁻⁴，βT=0.02，T=1000': r'$\beta_1=1\times 10^{-4}, \beta_T=0.02, T=1000$',
    'xₜ = √ᾱₜ · x₀ + √(1-ᾱₜ) · ε,    ε ~ N(0, I)': r'$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$',
    'ᾱₜ = ∏_{s=1}^t αₛ，αₜ = 1 - βₜ': r'$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s, \alpha_t = 1 - \beta_t$',
    'p_θ(xₜ₋₁ | xₜ, c) = N(xₜ₋₁; μ_θ(xₜ, t, c), σₜ²·I)': r'$p_\theta(x_{t-1} | x_t, c) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t, c), \sigma_t^2\mathbf{I})$',
    'μ_θ(xₜ, t, c) = (1/√αₜ) · [xₜ - (βₜ/√(1-ᾱₜ)) · ε_θ(xₜ, t, c)]': r'$\mu_\theta(x_t, t, c) = \frac{1}{\sqrt{\alpha_t}} \left[ x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t, c) \right]$',
    'ε_θ': r'$\epsilon_\theta$',
    'σₜ² = βₜ': r'$\sigma_t^2 = \beta_t$',
    'L = E_{x₀, ε, t} [ || ε - ε_θ(√ᾱₜ·x₀ + √(1-ᾱₜ)·ε, t, c) ||² ]': r'$L = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t, c) \|^2 \right]$',
    'Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V': r'$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V$',
    "x' = x + W_proj · Attention(Q, K, V)": r"$x' = x + W_{\text{proj}} \cdot \text{Attention}(Q, K, V)$",
    'h = SiLU( GroupNorm( Conv1d(x) ) ) + MLP(t_emb)[:, :, None]': r'$h = \text{SiLU}( \text{GroupNorm}( \text{Conv1d}(x) ) ) + \text{MLP}(t_{\text{emb}})[:, :, \text{None}]$',
    '输出 = SiLU( GroupNorm( Conv1d(h) ) ) + shortcut(x)': r'$\text{输出} = \text{SiLU}( \text{GroupNorm}( \text{Conv1d}(h) ) ) + \text{shortcut}(x)$',
    'N = 77.6P/T + 3.73×10⁵e/T²': r'$N = 77.6\frac{P}{T} + 3.73\times 10^5\frac{e}{T^2}$',
    '3.73×10⁵e/T²': r'$3.73\times 10^5\frac{e}{T^2}$',
    '150 K ≤ T ≤ 350 K': r'$150\text{ K} \leq T \leq 350\text{ K}$',
    '0.01 mb ≤ P ≤ 1100 mb': r'$0.01\text{ mb} \leq P \leq 1100\text{ mb}$',
    '≥ 0 km': r'$\geq 0\text{ km}$',
    '≥ 10 点': r'$\geq 10\text{ 点}$'
}

for old, new in replacements.items():
    text = text.replace(old, new)

with open(r'D:\02_Study\01_Schoolwork\Graduation Design\docs\midterm\林逸飞-220110814-本科毕业设计中期报告_latex.txt', 'w', encoding='utf-8') as f:
    f.write(text)
print("Done")
