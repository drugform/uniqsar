export interface Molecula {
  smiles: string;
  canonSmiles: string;
  /**
   * base64
   */
  binmol: string;
  imgurl: string;
  molmap: { x: number; y: number };
}

export interface ScoreInfo {
  value: number;
  score: number;
  report: string;
}

export interface MolCalc {
  logs: { water_solubility_light: ScoreInfo };
  o_mus_orl_ld: {
    toxicity_light: ScoreInfo;
  };
  total: { score: number };
}

export interface MolCalcResult {
  num: number;
  mol: Molecula;
  calcMol?: MolCalc;
}
