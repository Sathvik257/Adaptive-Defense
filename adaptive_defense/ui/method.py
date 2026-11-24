import pandas as pd
import streamlit as st


def render(run, summary, metrics, pred):
    st.subheader("How U-CADT Works")
    st.markdown(
        '<div class="pipeline"><span>Traffic Window</span><b>→</b><span>Uncertainty Autoencoder (MC Dropout)</span><b>→</b><span>Reconstruction Error & Epistemic Uncertainty</span><b>→</b><span>7-Factor Dynamic Modulation</span><b>→</b><span>Autonomous Alert</span></div>',
        unsafe_allow_html=True
    )
    
    st.markdown("### 1. Epistemic Uncertainty via Monte Carlo Dropout")
    st.write(
        "Standard autoencoders output only a point reconstruction error $e_t$. U-CADT utilizes Monte Carlo Dropout ($p=0.10$) "
        "across $M=5$ stochastic forward passes during inference to compute both mean error $e_t$ and epistemic uncertainty $\\mathcal{U}_t$. "
        "High uncertainty signals distribution shift or sensor noise, allowing the controller to regularize and avoid false-positive alert cascades."
    )

    st.markdown("### 2. Feedback-Free 8-Dimensional State Representation")
    st.write(
        "Unlike prior methods (e.g., Yang et al. 2023) that assume an omniscient human operator provides instantaneous TP/FP/TN/FN labels, "
        "U-CADT constructs an autonomous 8D state vector $s_t$ without requiring any test-time labels:"
    )
    st.latex(r"s_t = \left[ \mu_{e}, \sigma_{e}^2, \Delta e, \mu_{\mathcal{U}}, \sigma_{\mathcal{U}}^2, \frac{\delta_{t-1}}{2\theta_0}, \rho_{\text{alert}}, \text{CR}_t \right]")

    st.markdown("### 3. Continuous 7-Factor Threshold Modulation")
    st.write(
        "Rather than switching abruptly between binary $\{0, 1\}$ states, U-CADT adaptively scales an extreme-value calibrated base threshold $\\theta_0$ "
        "across 7 fine-grained quantile offsets:"
    )
    st.latex(r"\mathcal{A} = \{-0.25, -0.15, -0.05, 0.0, +0.05, +0.15, +0.25\}, \quad \delta_t = \theta_0 (1 + a_t)")

    st.markdown("### 4. Stability-Regularized Objective")
    st.write(
        "To prevent chattering and false alarms under non-stationary traffic, the reinforcement learning policy penalizes threshold volatility:"
    )
    st.latex(r"r_t = 2.0 \cdot \text{TP} + 0.5 \cdot \text{TN} - 2.0 \cdot \text{FP} - 2.0 \cdot \text{FN} - \lambda_{\text{vol}} \left(\frac{\Delta \delta}{\theta_0}\right)^2")

    if "limitations" in summary:
        st.subheader("Design Constraints & Limitations")
        for limitation in summary["limitations"]:
            st.write("• " + limitation)
    
    with st.expander("Experiment configuration"):
        st.json(summary)
