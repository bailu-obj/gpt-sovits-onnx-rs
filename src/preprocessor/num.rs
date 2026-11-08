// preprocessor/num.rs
//
// Chinese number expansion via pest grammar (`resource/rule.pest`).
// Unit-suffixed patterns (年/元/%) are handled by regex in `text_normalize.rs`.

use anyhow::{Result, bail};
use pest::Parser;

#[derive(pest_derive::Parser)]
#[grammar = "resource/rule.pest"]
pub struct ExprParser;

pub mod zh {
    use super::*;
    use pest::iterators::Pair;

    fn parse_pn(pair: Pair<Rule>, dst_string: &mut String) -> Result<()> {
        assert_eq!(pair.as_rule(), Rule::pn);
        match pair.as_str() {
            "+" => dst_string.push_str("加"),
            "-" => dst_string.push_str("减"),
            "*" | "×" => dst_string.push_str("乘"),
            "/" | "÷" => dst_string.push_str("除以"),
            "=" => dst_string.push_str("等于"),
            _ => bail!("Unknown operator: {:?}", pair.as_str()),
        }
        Ok(())
    }

    fn parse_flag(pair: Pair<Rule>, dst_string: &mut String) -> Result<()> {
        assert_eq!(pair.as_rule(), Rule::flag);
        match pair.as_str() {
            "+" => dst_string.push_str("正"),
            "-" => dst_string.push_str("负"),
            _ => bail!("Unknown flag: {:?}", pair.as_str()),
        }
        Ok(())
    }

    fn parse_percent(pair: Pair<Rule>, dst_string: &mut String) -> Result<()> {
        assert_eq!(pair.as_rule(), Rule::percent);
        dst_string.push_str("百分之");
        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::decimals => parse_decimals(pair, dst_string)?,
                Rule::integer => parse_integer(pair, dst_string, true)?,
                _ => bail!("Unknown rule in percent: {:?}", pair.as_str()),
            }
        }
        Ok(())
    }

    const DIGITS: [char; 10] = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
    const UNITS: [(usize, char); 5] = [(8, '亿'), (4, '万'), (3, '千'), (2, '百'), (1, '十')];

    fn append_cardinal(value: &str, output: &mut String) {
        let significant = value.trim_start_matches('0');
        match significant.len() {
            0 => {}
            1 => {
                if significant.len() < value.len() {
                    output.push('零');
                }
                output.push(DIGITS[(significant.as_bytes()[0] - b'0') as usize]);
            }
            _ => {
                let (power, unit) = UNITS
                    .iter()
                    .copied()
                    .find(|&(power, _)| power < significant.len())
                    .unwrap();
                let (head, tail) = value.split_at(value.len() - power);
                append_cardinal(head, output);
                output.push(unit);
                append_cardinal(tail, output);
            }
        }
    }

    fn parse_integer(pair: Pair<Rule>, dst_string: &mut String, unit: bool) -> Result<()> {
        assert_eq!(pair.as_rule(), Rule::integer);
        let value = pair.as_str();
        if !unit {
            dst_string.extend(value.bytes().map(|byte| DIGITS[(byte - b'0') as usize]));
            return Ok(());
        }

        let value = value.trim_start_matches('0');
        if value.is_empty() {
            dst_string.push('零');
            return Ok(());
        }
        let mut cardinal = String::new();
        append_cardinal(value, &mut cardinal);
        dst_string.push_str(if cardinal.starts_with("一十") {
            cardinal.strip_prefix('一').unwrap()
        } else {
            &cardinal
        });
        Ok(())
    }

    fn parse_decimals(pair: Pair<Rule>, dst_string: &mut String) -> Result<()> {
        assert_eq!(pair.as_rule(), Rule::decimals);

        let mut inner = pair.into_inner().rev();
        let f_part = inner.next().unwrap();
        if let Some(i_part) = inner.next() {
            parse_integer(i_part, dst_string, true)?;
        } else {
            dst_string.push_str("零");
        }
        dst_string.push_str("点");
        parse_integer(f_part, dst_string, false)?;

        Ok(())
    }

    fn parse_fractional(pair: Pair<Rule>, dst_string: &mut String) -> Result<()> {
        assert_eq!(pair.as_rule(), Rule::fractional);

        let mut inner = pair.into_inner();
        let numerator = inner.next().unwrap();
        let denominator = inner.next().unwrap();
        parse_integer(denominator, dst_string, true)?;
        dst_string.push_str("分之");
        parse_integer(numerator, dst_string, true)?;
        Ok(())
    }

    fn parse_num(pair: Pair<Rule>, dst_string: &mut String) -> Result<()> {
        assert_eq!(pair.as_rule(), Rule::num);

        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::flag => parse_flag(pair, dst_string)?,
                Rule::percent => parse_percent(pair, dst_string)?,
                Rule::decimals => parse_decimals(pair, dst_string)?,
                Rule::fractional => parse_fractional(pair, dst_string)?,
                Rule::integer => parse_integer(pair, dst_string, true)?,
                _ => bail!("Unknown rule in num: {:?}", pair.as_str()),
            }
        }
        Ok(())
    }

    fn parse_signs(pair: Pair<Rule>, dst_string: &mut String) -> Result<()> {
        assert_eq!(pair.as_rule(), Rule::signs);

        let inner = pair.into_inner();
        for pair in inner {
            log::debug!("{:?}", pair);
            match pair.as_rule() {
                Rule::num => parse_num(pair, dst_string)?,
                Rule::pn => parse_pn(pair, dst_string)?,
                Rule::word => {
                    log::warn!("word: {:?}", pair.as_str());
                }
                _ => bail!("Unknown rule in signs: {:?}", pair.as_str()),
            }
        }
        Ok(())
    }

    fn parse_link(pair: Pair<Rule>, dst_string: &mut String) -> Result<()> {
        assert_eq!(pair.as_rule(), Rule::link);
        if pair.as_str() == "-" {
            dst_string.push_str("杠");
        }
        Ok(())
    }

    fn parse_word(pair: Pair<Rule>, dst_string: &mut String) -> Result<()> {
        assert_eq!(pair.as_rule(), Rule::word);
        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::digit => {
                    let txt = match pair.as_str() {
                        "0" => "零",
                        "1" => "一",
                        "2" => "二",
                        "3" => "三",
                        "4" => "四",
                        "5" => "五",
                        "6" => "六",
                        "7" => "七",
                        "8" => "八",
                        "9" => "九",
                        _ => bail!("Unknown digit: {:?}", pair.as_str()),
                    };
                    dst_string.push_str(txt);
                }
                Rule::alpha => {
                    dst_string.push_str(pair.as_str());
                }
                Rule::greek => {
                    let txt = match pair.as_str() {
                        "α" | "Α" => "阿尔法",
                        "β" | "Β" => "贝塔",
                        "γ" | "Γ" => "伽马",
                        "δ" | "Δ" => "德尔塔",
                        "ε" | "Ε" => "艾普西龙",
                        "ζ" | "Ζ" => "泽塔",
                        "η" | "Η" => "艾塔",
                        "θ" | "Θ" => "西塔",
                        "ι" | "Ι" => "约塔",
                        "κ" | "Κ" => "卡帕",
                        "λ" | "Λ" => "兰姆达",
                        "μ" | "Μ" => "缪",
                        "ν" | "Ν" => "纽",
                        "ξ" | "Ξ" => "克西",
                        "ο" | "Ο" => "欧米克戈",
                        "π" | "Π" => "派",
                        "ρ" | "Ρ" => "罗",
                        "σ" | "Σ" => "西格玛",
                        "τ" | "Τ" => "套",
                        "υ" | "Υ" => "宇普西龙",
                        "φ" | "Φ" => "斐",
                        "χ" | "Χ" => "希",
                        "ψ" | "Ψ" => "普西",
                        "ω" | "Ω" => "欧米伽",
                        _ => bail!("Unknown Greek letter: {:?}", pair.as_str()),
                    };
                    dst_string.push_str(txt);
                }
                _ => bail!("Unknown rule in word: {:?}", pair.as_str()),
            }
        }
        Ok(())
    }

    fn parse_ident(pair: Pair<Rule>, dst_string: &mut String) -> Result<()> {
        assert_eq!(pair.as_rule(), Rule::ident);
        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::word => parse_word(pair, dst_string)?,
                Rule::link => parse_link(pair, dst_string)?,
                _ => bail!("Unknown rule in ident: {:?}", pair.as_str()),
            }
        }
        Ok(())
    }

    pub fn parse_all(pair: Pair<Rule>, dst_string: &mut String) -> Result<()> {
        assert_eq!(pair.as_rule(), Rule::all);
        let inner = pair.into_inner();
        for pair in inner {
            match pair.as_rule() {
                Rule::signs => parse_signs(pair, dst_string)?,
                Rule::ident => parse_ident(pair, dst_string)?,
                _ => bail!("Unknown rule in all: {:?}", pair.as_str()),
            }
        }
        Ok(())
    }
}

/// Expand a bare numeric token to Chinese words (pest grammar).
pub fn expand_zh(token: &str) -> String {
    NumSentence {
        text: token.to_string(),
    }
    .to_lang_text()
    .unwrap_or_else(|_| token.to_string())
}

#[derive(Debug)]
pub struct NumSentence {
    pub text: String,
}

impl NumSentence {
    pub fn to_lang_text(&self) -> Result<String> {
        let mut dst_string = String::new();
        let pairs = ExprParser::parse(Rule::all, &self.text)?;
        for pair in pairs {
            zh::parse_all(pair, &mut dst_string)?;
        }
        Ok(dst_string.trim().to_string())
    }
}
