import streamlit as st
import pandas as pd
import torch
from pathlib import Path
from chemprop import data, featurizers, models
import lightning.pytorch as pl
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np

def draw_mol(m, highlights=None, size=(400, 400)):
    if m is None:
        return None
    mc = Chem.Mol(m)
    if highlights:
        for i, idx in enumerate(highlights, 1):
            mc.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(i))
    opts = Draw.MolDrawOptions()
    opts.includeAtomNumbers = False
    opts.addAtomIndices = False
    opts.bondLineWidth = 2
    opts.fixedBondLength = 30
    opts.annotationFontScale = 0.7
    return Draw.MolToImage(mc, size=size, legend="", highlightAtoms=highlights, options=opts)

def gen_pa(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return []
    inputs = []
    for atom in m.GetAtoms():
        if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0:
            idx = atom.GetIdx()
            nm = Chem.Mol(m)
            nm.GetAtomWithIdx(idx).SetNumExplicitHs(atom.GetTotalNumHs() - 1)
            nm.GetAtomWithIdx(idx).SetFormalCharge(-1)
            Chem.SanitizeMol(nm)
            inputs.append((idx, Chem.MolToSmiles(nm, isomericSmiles=True, canonical=True)))
    return inputs

def gen_bde(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return []
    inputs = []
    for atom in m.GetAtoms():
        if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0:
            idx = atom.GetIdx()
            nm = Chem.RWMol(m)
            a = nm.GetAtomWithIdx(idx)
            a.SetNumExplicitHs(a.GetTotalNumHs() - 1)
            a.SetNoImplicit(True)
            a.SetNumRadicalElectrons(1)
            a.SetFormalCharge(0)
            Chem.SanitizeMol(nm, Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
            inputs.append((idx, Chem.MolToSmiles(nm, isomericSmiles=True, canonical=True)))
    return inputs

@st.cache_resource
def get_model(ckpt_path):
    return models.MPNN.load_from_checkpoint(ckpt_path)

def run_pred(model, smis):
    if not smis: return np.array([])
    f = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dps = [data.MoleculeDatapoint.from_smi(s) for s in smis]
    ds = data.MoleculeDataset(dps, featurizer=f)
    dl = data.build_dataloader(ds, shuffle=False, num_workers=0)
    trainer = pl.Trainer(logger=None, enable_progress_bar=False, accelerator="cpu", devices=1)
    preds = trainer.predict(model, dl)
    return np.concatenate(preds, axis=0)

def main():
    st.set_page_config(page_title="Antioxidant Property Predictor", layout="wide")

    with st.sidebar:
        st.header("About This App")
        st.info(
            "This application uses Graph Neural Network (GNN) models to predict "
            "antioxidant properties (IP, BDE, PA) from a molecule's SMILES string."
        )
        st.markdown(
            """
            ---
            **How to Cite**  
            If you use this tool in your research, please cite our manuscript:

            > Z. M. Wong; D. W. P. Tay; Y. H. Lim; S. J. Ang. (2025). *An Iterative DFT and Machine Learning Strategy for High-Throughput Antioxidant Screening of Flavan-3-ols*.  
            > *Manuscript Submitted*  
            > [DOI Link Here](https://doi.org/your-doi-link)

            **Source Code**  
            The code for this application is available on [GitHub](https://github.com/omarvino/antioxidant).
            """
        )
    
    st.title("🧪 Antioxidant Property Predictor")
    st.markdown(
        """
        Enter a molecule's SMILES string to predict its antioxidant properties. All values are in **kcal/mol**.
        - **IP**: Ionization Potential
        - **BDE**: O-H Bond Dissociation Energy
        - **PA**: Proton Affinity
        """
    )
    
    p = Path("./checkpoints")
    p.mkdir(exist_ok=True)
    ckpt_paths = {
        "IP": p / "ip_model.ckpt", "BDE_site": p / "bde_site_model.ckpt",
        "PA_site": p / "pa_site_model.ckpt", "BDE_mol": p / "bde_mol_model.ckpt",
        "PA_mol": p / "pa_mol_model.ckpt",
    }
    for path in ckpt_paths.values():
        path.touch()

    smi_in = st.text_input(
        "Enter SMILES string:", "Oc1ccc(cc1O)[C@H]3Oc2cc(O)cc(O)c2C[C@@H]3O", key="smiles_input"
    )

    if st.button("Predict Properties", type="primary"):
        if not smi_in.strip():
            st.error("Please enter a valid SMILES string.")
            return

        mol = Chem.MolFromSmiles(smi_in)
        if mol is None:
            st.error("Invalid SMILES string provided. Please check the format.")
            return

        with st.spinner("Analyzing molecule and running predictions..."):
            oh_sites = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'O' and a.GetTotalNumHs() > 0]
            
            bde_in = gen_bde(smi_in)
            pa_in = gen_pa(smi_in)
            
            bde_idx, bde_smi = zip(*bde_in) if bde_in else ([], [])
            pa_idx, pa_smi = zip(*pa_in) if pa_in else ([], [])

            try:
                m_ip = get_model(ckpt_paths["IP"])
                m_bde_s = get_model(ckpt_paths["BDE_site"])
                m_pa_s = get_model(ckpt_paths["PA_site"])
                m_bde_m = get_model(ckpt_paths["BDE_mol"])
                m_pa_m = get_model(ckpt_paths["PA_mol"])

                p_ip = run_pred(m_ip, [smi_in])
                p_bde_m = run_pred(m_bde_m, [smi_in])
                p_pa_m = run_pred(m_pa_m, [smi_in])
                p_bde_s = run_pred(m_bde_s, list(bde_smi))
                p_pa_s = run_pred(m_pa_s, list(pa_smi))
            except Exception as e:
                st.error(f"Failed to load a model or run prediction. Please check checkpoint paths. Error: {e}")
                return

            res_bde = {idx: pred for idx, pred in zip(bde_idx, p_bde_s.flatten())}
            res_pa = {idx: pred for idx, pred in zip(pa_idx, p_pa_s.flatten())}

        col1, col2 = st.columns([2, 3], gap="large")

        with col1:
            st.subheader("Molecule Visualization")
            st.markdown("Hydroxyl (-OH) groups for site-specific prediction are numbered.")
            mol_img = draw_mol(mol, highlights=oh_sites, size=(500, 500))
            st.image(mol_img, use_container_width=True)

        with col2:
            st.subheader("Prediction Results")
            tab1, tab2, tab3 = st.tabs(["**Ionization Potential (IP)**", "**Bond Dissociation (BDE)**", "**Proton Affinity (PA)**"])

            with tab1:
                st.metric(label="Predicted Ionization Potential", value=f"{p_ip[0][0]:.2f} kcal/mol")
                st.info("Lower IP values indicate the molecule is more easily oxidized (donates an electron).")

            with tab2:
                if not res_bde:
                    st.warning("No O-H bonds found for BDE prediction.")
                else:
                    st.metric(label="Molecule-Level BDE (Predicted Lowest)", value=f"{p_bde_m[0][0]:.2f} kcal/mol")
                    st.divider()
                    st.write("**Site-Specific Breakdown**")
                    min_site = min(res_bde, key=res_bde.get)
                    min_val = res_bde[min_site]
                    st.success(f"**Most Favorable Site:** Site #{oh_sites.index(min_site) + 1} with a BDE of **{min_val:.2f} kcal/mol**.")
                    st.info("Lower BDE indicates a weaker O-H bond, making hydrogen atom transfer more favorable.")
                    df = pd.DataFrame([(oh_sites.index(idx) + 1, val) for idx, val in res_bde.items()], columns=["Site #", "BDE (kcal/mol)"]).sort_values(by="Site #").set_index("Site #")
                    st.dataframe(df.style.format("{:.2f}"))

            with tab3:
                if not res_pa:
                    st.warning("No O-H bonds found for PA prediction.")
                else:
                    st.metric(label="Molecule-Level PA (Predicted Lowest)", value=f"{p_pa_m[0][0]:.2f} kcal/mol")
                    st.divider()
                    st.write("**Site-Specific Breakdown**")
                    min_site = min(res_pa, key=res_pa.get)
                    min_val = res_pa[min_site]
                    st.success(f"**Most Favorable Site:** Site #{oh_sites.index(min_site) + 1} with a PA of **{min_val:.2f} kcal/mol**.")
                    st.info("Lower PA indicates the corresponding anion is a weaker base, making deprotonation more favorable.")
                    df = pd.DataFrame([(oh_sites.index(idx) + 1, val) for idx, val in res_pa.items()], columns=["Site #", "PA (kcal/mol)"]).sort_values(by="Site #").set_index("Site #")
                    st.dataframe(df.style.format("{:.2f}"))
        
        st.divider()
        st.subheader("Downloadable Results Summary")
        summary_data = [
            {"Property": "IP", "Type": "Molecular", "Site": "-", "Value (kcal/mol)": f"{p_ip[0][0]:.2f}"},
            {"Property": "BDE", "Type": "Molecular", "Site": "-", "Value (kcal/mol)": f"{p_bde_m[0][0]:.2f}"},
            {"Property": "PA", "Type": "Molecular", "Site": "-", "Value (kcal/mol)": f"{p_pa_m[0][0]:.2f}"}
        ]
        summary_data.extend([{"Property": "BDE", "Type": "Site-Specific", "Site": str(oh_sites.index(idx) + 1), "Value (kcal/mol)": f"{val:.2f}"} for idx, val in res_bde.items()])
        summary_data.extend([{"Property": "PA", "Type": "Site-Specific", "Site": str(oh_sites.index(idx) + 1), "Value (kcal/mol)": f"{val:.2f}"} for idx, val in res_pa.items()])
        
        df_final = pd.DataFrame(summary_data)
        st.dataframe(df_final, use_container_width=True)        
        st.download_button(
            label="Download All Results as CSV",
            data=df_final.to_csv(index=False).encode('utf-8'),
            file_name=f"predictions_{smi_in}.csv",
            mime='text/csv',
        )

if __name__ == "__main__":
    main()
