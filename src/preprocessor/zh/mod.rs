// text/zh/mod.rs
use log::debug;

use crate::preprocessor::zh::{g2pw::G2PW, mandarin_g2p::G2pResult};
pub mod g2pw;
mod jyutping_list;
pub mod mandarin_g2p;
pub mod split;
pub mod tone_sandhi;
pub mod yue;

#[derive(Debug)]
pub enum ZhMode {
    Mandarin,
    Cantonese,
}

#[derive(Debug, Default)]
pub struct ZhSentence {
    pub phones: Vec<String>,
    pub word2ph: Vec<i32>,
    pub text: String,
}

impl ZhSentence {
    pub fn g2p(&mut self, g2pw: &mut G2PW, jieba: &jieba_rs::Jieba, mode: ZhMode) {
        match mode {
            ZhMode::Mandarin => self.apply_g2p(mandarin_g2p::g2p_mandarin(&self.text, g2pw, jieba)),
            ZhMode::Cantonese => {
                let (phones, word2ph) = yue::g2p(&self.text);
                debug!("Cantonese G2P for '{}': {:?}", self.text, phones);
                self.phones = phones;
                self.word2ph = word2ph;
            }
        }
    }

    fn apply_g2p(&mut self, result: G2pResult) {
        self.phones = result.phones;
        self.word2ph = result.word2ph;
        debug!(
            "Mandarin G2P for '{}': {} phones",
            self.text,
            self.phones.len()
        );
    }
}
