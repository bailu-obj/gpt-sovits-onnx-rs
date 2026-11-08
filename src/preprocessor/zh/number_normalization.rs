// Ordered Mandarin numeric verbalization following GPT-SoVITS TextNormalizer.
use once_cell::sync::Lazy;
use regex::{Captures, Regex};

use crate::preprocessor::num::expand_zh;

const NUMBER: &str = r"-?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)";
const CLOCK: &str = r"(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?";
const MEASURES: &[(&str, &str)] = &[
    ("cm2", "平方厘米"),
    ("cm²", "平方厘米"),
    ("cm3", "立方厘米"),
    ("cm³", "立方厘米"),
    ("mm", "毫米"),
    ("cm", "厘米"),
    ("kg", "千克"),
    ("km", "千米"),
    ("ml", "毫升"),
    ("m2", "平方米"),
    ("m²", "平方米"),
    ("m3", "立方米"),
    ("m³", "立方米"),
    ("db", "分贝"),
    ("ds", "毫秒"),
    ("m", "米"),
    ("s", "秒"),
];
const QUANTIFIERS: &str = r"(?:平方厘米|立方厘米|平方米|立方米|分贝|毫秒|小时|(?:[千毫微]克)|(?:[千分厘毫微]米)|(?:亿|千万|百万|万|千|百|十)?吨|(?:亿|千万|百万|万|千|百|美)?元|(?:亿|千万|百万|万|千|百)?块|[处台架枚趟幅平方堵间床株批项例列篇栋注亩封艘把目套段人所朵匹张座回场尾条个首阙阵网炮顶丘棵只支袭辆挑担颗壳窠曲墙群腔砣客贯扎捆刀令打手罗坡山岭江溪钟队单双对出口头脚板跳枝件贴针线管名位身堂课本页家户层丝毫厘分钱两斤铢石钧锱忽寸尺丈里寻常铺程米撮勺合升斗盘碗碟叠桶笼盆盒杯斛锅簋篮罐瓶壶卮盏箩箱煲啖袋钵年月日号季刻时周天秒旬纪岁世更夜春夏秋冬代伏辈丸泡粒幢堆根道面片角毛亿万千百])";

fn regex(pattern: &str) -> Regex {
    Regex::new(pattern).expect("valid Mandarin normalization regex")
}

static DATE: Lazy<Regex> =
    Lazy::new(|| regex(r"([0-9]{4})([-/. ])(0[1-9]|1[0-2])([-/. ])(0[1-9]|[12][0-9]|3[01])"));
static YEAR: Lazy<Regex> = Lazy::new(|| regex(r"([0-9]+)年"));
static TIME_RANGE: Lazy<Regex> = Lazy::new(|| regex(&format!(r"({CLOCK})[-~]({CLOCK})")));
static TIME: Lazy<Regex> = Lazy::new(|| regex(CLOCK));
static TEMPERATURE: Lazy<Regex> = Lazy::new(|| regex(&format!(r"({NUMBER})(摄氏度|°C|℃|度)")));
static QUALIFIED_RANGE: Lazy<Regex> = Lazy::new(|| {
    let units = measure_pattern();
    let suffix = format!(r"(?:%|摄氏度|°C|℃|度|{units})");
    regex(&format!(r"({NUMBER}{suffix})~({NUMBER}{suffix})"))
});
static MEASURE: Lazy<Regex> = Lazy::new(|| regex(&format!(r"([0-9])({})", measure_pattern())));
static MOBILE: Lazy<Regex> =
    Lazy::new(|| regex(r"(?:\+?86 *)?1(?:[38][0-9]|5[0-35-9]|7[678]|9[89])[0-9]{8}"));
static LANDLINE: Lazy<Regex> =
    Lazy::new(|| regex(r"(?:0(?:10|2[1-3]|[3-9][0-9]{2})-?)?[1-9][0-9]{6,7}"));
static SERVICE_NUMBER: Lazy<Regex> = Lazy::new(|| regex(r"400-?[0-9]{3}-?[0-9]{4}"));
static FRACTION: Lazy<Regex> = Lazy::new(|| regex(r"(-?)([0-9]+)/([0-9]+)"));
static PERCENT: Lazy<Regex> = Lazy::new(|| regex(r"(-?)([0-9]+(?:\.[0-9]+)?)%"));
const SUPERSCRIPTS: &str = "⁰¹²³⁴⁵⁶⁷⁸⁹ˣʸⁿ";
static POWER: Lazy<Regex> = Lazy::new(|| regex(&format!(r"[{SUPERSCRIPTS}]+")));
static VERSION: Lazy<Regex> = Lazy::new(|| regex(r"[0-9]+(?:\.[0-9]+){2,}"));
static RANGE: Lazy<Regex> = Lazy::new(|| regex(&format!(r"({NUMBER})~({NUMBER})")));
static DECIMAL: Lazy<Regex> = Lazy::new(|| regex(r"-?[0-9]*\.[0-9]+"));
static NEGATIVE: Lazy<Regex> = Lazy::new(|| regex(r"-[0-9]+"));
static QUANTITY: Lazy<Regex> =
    Lazy::new(|| regex(&format!(r"([0-9]+)([多余几+]?)({QUANTIFIERS})")));
static INTEGER: Lazy<Regex> = Lazy::new(|| regex(r"[0-9]+"));
static MATH_OPERATOR: Lazy<Regex> = Lazy::new(|| regex(r"[+×÷=]"));

fn measure_pattern() -> String {
    MEASURES
        .iter()
        .map(|(unit, _)| *unit)
        .collect::<Vec<_>>()
        .join("|")
}

pub(in crate::preprocessor) fn is_numeric_unit(text: &str) -> bool {
    matches!(text, "°C" | "℃") || MEASURES.iter().any(|(unit, _)| *unit == text)
}

pub(in crate::preprocessor) fn spoken_symbol(c: char) -> Option<&'static str> {
    Some(match c {
        '①' => "一",
        '②' => "二",
        '③' => "三",
        '④' => "四",
        '⑤' => "五",
        '⑥' => "六",
        '⑦' => "七",
        '⑧' => "八",
        '⑨' => "九",
        '⑩' => "十",
        'α' => "阿尔法",
        'β' => "贝塔",
        'γ' | 'Γ' => "伽玛",
        'δ' | 'Δ' => "德尔塔",
        'ε' => "艾普西龙",
        'ζ' => "捷塔",
        'η' => "依塔",
        'θ' | 'Θ' => "西塔",
        'ι' => "艾欧塔",
        'κ' => "喀帕",
        'λ' | 'Λ' => "拉姆达",
        'μ' => "缪",
        'ν' => "拗",
        'ξ' | 'Ξ' => "克西",
        'ο' => "欧米克伦",
        'π' | 'Π' => "派",
        'ρ' => "肉",
        'ς' | 'Σ' | 'σ' => "西格玛",
        'τ' => "套",
        'υ' => "宇普西龙",
        'φ' | 'Φ' => "服艾",
        'χ' => "器",
        'ψ' | 'Ψ' => "普赛",
        'ω' | 'Ω' => "欧米伽",
        _ => return None,
    })
}

pub(in crate::preprocessor) fn normalize_numbers(text: &str) -> String {
    let text = bounded_replace(&DATE, text, |c| {
        if c[2] != c[4] {
            return c[0].to_owned();
        }
        format!(
            "{}年{}月{}日",
            digits(&c[1], false),
            expand_zh(&c[3]),
            expand_zh(&c[5])
        )
    });
    let text = YEAR.replace_all(&text, |c: &Captures<'_>| {
        if matches!(c[1].len(), 2 | 4) {
            format!("{}年", digits(&c[1], false))
        } else {
            c[0].to_owned()
        }
    });
    let text = bounded_replace(&TIME_RANGE, &text, |c| {
        format!("{}至{}", time(&c[1]), time(&c[2]))
    });
    let text = bounded_replace(&TIME, &text, |c| time(&c[0]));
    let text = QUALIFIED_RANGE.replace_all(&text, "${1}至${2}");
    let text = TEMPERATURE.replace_all(&text, |c: &Captures<'_>| {
        let (sign, number) = c[1].strip_prefix('-').map_or(("", &c[1]), |n| ("零下", n));
        let unit = if &c[2] == "摄氏度" {
            "摄氏度"
        } else {
            "度"
        };
        format!("{sign}{}{unit}", number_text(number))
    });
    let text = MEASURE.replace_all(&text, |c: &Captures<'_>| {
        let unit = MEASURES.iter().find(|(unit, _)| *unit == &c[2]).unwrap().1;
        format!("{}{unit}", &c[1])
    });
    let text = bounded_replace(&MOBILE, &text, |c| phone(&c[0]));
    let text = bounded_replace(&SERVICE_NUMBER, &text, |c| phone(&c[0]));
    let text = bounded_replace(&LANDLINE, &text, |c| phone(&c[0]));
    let text = FRACTION.replace_all(&text, |c: &Captures<'_>| {
        format!(
            "{}{}分之{}",
            minus(&c[1]),
            expand_zh(&c[3]),
            expand_zh(&c[2])
        )
    });
    let text = PERCENT.replace_all(&text, |c: &Captures<'_>| {
        format!("{}百分之{}", minus(&c[1]), number_text(&c[2]))
    });
    let text = RANGE.replace_all(&text, |c: &Captures<'_>| {
        format!("{}到{}", number_text(&c[1]), number_text(&c[2]))
    });
    let text = VERSION.replace_all(&text, |c: &Captures<'_>| {
        c[0].split('.')
            .map(|part| digits(part, false))
            .collect::<Vec<_>>()
            .join("点")
    });
    let text = normalize_operators(&text);
    let text = POWER.replace_all(&text, |c: &Captures<'_>| {
        let exponent: String = c[0]
            .chars()
            .filter_map(|c| {
                SUPERSCRIPTS
                    .chars()
                    .position(|symbol| symbol == c)
                    .map(|index| b"0123456789xyn"[index] as char)
            })
            .collect();
        format!("的{exponent}次方")
    });
    let text = DECIMAL.replace_all(&text, |c: &Captures<'_>| number_text(&c[0]));
    let text = NEGATIVE.replace_all(&text, |c: &Captures<'_>| number_text(&c[0]));
    let text = QUANTITY.replace_all(&text, |c: &Captures<'_>| {
        let amount = expand_zh(&c[1]);
        let amount = if amount == "二" && !matches!(&c[3], "年" | "月" | "日" | "号") {
            "两"
        } else {
            &amount
        };
        let extra = if &c[2] == "+" { "多" } else { &c[2] };
        format!("{amount}{extra}{}", &c[3])
    });
    let text = INTEGER.replace_all(&text, |c: &Captures<'_>| {
        if c[0].len() >= 3 {
            digits(&c[0], true)
        } else {
            expand_zh(&c[0])
        }
    });
    MATH_OPERATOR
        .replace_all(&text, |c: &Captures<'_>| match &c[0] {
            "+" => "加",
            "×" => "乘",
            "÷" => "除",
            _ => "等",
        })
        .replace('/', "每")
}

// Regex does not support lookaround. Keep the number matches out of longer digit runs.
fn bounded_replace(
    pattern: &Regex,
    text: &str,
    replace: impl Fn(&Captures<'_>) -> String,
) -> String {
    pattern
        .replace_all(text, |c: &Captures<'_>| {
            let matched = c.get(0).unwrap();
            let bytes = text.as_bytes();
            if matched
                .start()
                .checked_sub(1)
                .is_some_and(|i| bytes[i].is_ascii_digit())
                || bytes.get(matched.end()).is_some_and(u8::is_ascii_digit)
            {
                matched.as_str().to_owned()
            } else {
                replace(c)
            }
        })
        .into_owned()
}

fn normalize_operators(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut previous = None;
    while let Some(c) = chars.next() {
        let between_numbers = previous
            .is_some_and(|c: char| c.is_ascii_digit() || SUPERSCRIPTS.contains(c))
            && chars
                .peek()
                .is_some_and(|c| c.is_ascii_digit() || matches!(c, '-' | '.'));
        match c {
            '-' if between_numbers => result.push('减'),
            '+' if between_numbers => result.push('加'),
            '=' if between_numbers => result.push_str("等于"),
            '×' if between_numbers => result.push('乘'),
            '÷' if between_numbers => result.push('除'),
            _ => result.push(c),
        }
        previous = Some(c);
    }
    result
}

fn time(value: &str) -> String {
    let mut fields = value.split(':');
    let hour = fields.next().unwrap();
    let minute = fields.next().unwrap();
    let mut result = format!("{}点", expand_zh(hour));
    if minute == "30" {
        result.push('半');
    } else {
        append_clock_field(&mut result, minute, '分');
    }
    if let Some(second) = fields.next() {
        append_clock_field(&mut result, second, '秒');
    }
    result
}

fn append_clock_field(result: &mut String, field: &str, unit: char) {
    if field != "00" {
        if field.starts_with('0') {
            result.push('零');
        }
        result.push_str(&expand_zh(field));
        result.push(unit);
    }
}

fn phone(value: &str) -> String {
    value
        .trim_start_matches('+')
        .split([' ', '-'])
        .filter(|part| !part.is_empty())
        .map(|part| digits(part, true))
        .collect::<Vec<_>>()
        .join(",")
}

fn minus(sign: &str) -> &str {
    if sign == "-" { "负" } else { "" }
}

fn number_text(token: &str) -> String {
    let (sign, token) = token.strip_prefix('-').map_or(("", token), |n| ("负", n));
    let Some((integer, fraction)) = token.split_once('.') else {
        return format!("{sign}{}", expand_zh(token));
    };
    let integer = if integer.is_empty() { "0" } else { integer };
    let mut result = format!("{sign}{}点", expand_zh(integer));
    result.push_str(&digits(fraction.trim_end_matches('0'), false));
    if fraction.ends_with('0') {
        result.push('零');
    }
    result
}

fn digits(value: &str, alternate_one: bool) -> String {
    const DIGITS: [char; 10] = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
    value
        .bytes()
        .filter(u8::is_ascii_digit)
        .map(|d| {
            if alternate_one && d == b'1' {
                '幺'
            } else {
                DIGITS[(d - b'0') as usize]
            }
        })
        .collect()
}
