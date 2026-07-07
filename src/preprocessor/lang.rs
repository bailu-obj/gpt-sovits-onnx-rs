// preprocessor/lang.rs
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lang {
    Zh,
    En,
}

impl Default for Lang {
    fn default() -> Self {
        Lang::Zh
    }
}

#[derive(Debug, Clone, Copy)]
pub enum LangId {
    Auto,    // Mandarin
    AutoYue, // Cantonese
}
