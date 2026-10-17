// Циклические алгоритмы с carried-состоянием — вход для batched-VM.
// Всё на i32/u8, без аллокаций: память wasm используется как
// read-only вход (input, таблицы) + write-only выход (stores логируем).

static STEP_TABLE: [i32; 89] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
    19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
    130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
    337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
    876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
    2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
    5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
    15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
];

static INDEX_TABLE: [i32; 8] = [-1, -1, -1, -1, 2, 4, 6, 8];

/// IMA ADPCM: состояние (pred, index) carried через весь поток.
/// В лоб не параллелится, но ошибки состояния затухают:
/// index гуляет по лесенке и сливается на клампах 0/88,
/// pred — «дырявый» интегратор, чей след от начальной ошибки тает.
#[no_mangle]
pub extern "C" fn adpcm_decode(input: *const u8, n_bytes: usize, out: *mut i16) -> usize {
    let mut pred: i32 = 0;
    let mut index: i32 = 0;
    let total = n_bytes * 2;
    let mut o: usize = 0;
    while o < total {
        let byte = unsafe { *input.add(o >> 1) } as i32;
        let nib = (byte >> (((o & 1) as i32) * 4)) & 15;
        let step = unsafe { *STEP_TABLE.as_ptr().add(index as usize) };
        let mut diff = step >> 3;
        diff += ((nib & 4) >> 2) * step;
        diff += ((nib & 2) >> 1) * (step >> 1);
        diff += (nib & 1) * (step >> 2);
        let signed = 0i32.wrapping_sub(diff) & (0i32.wrapping_sub((nib >> 3) & 1));
        pred += diff + signed * 2;
        pred = pred.clamp(-32768, 32767);
        index += unsafe { *INDEX_TABLE.as_ptr().add((nib & 7) as usize) };
        index = index.clamp(0, 88);
        unsafe { *out.add(o) = pred as i16 };
        o += 1;
    }
    total
}

/// G.726/G.722-стиль leaky ADPCM: оба состояния (pred, index) имеют
/// утечку 15/16 (<1 по спектру Якобиана) -> любая ошибка начального
/// состояния затухает до точного целочисленного нуля за ~180–240 сэмплов.
#[no_mangle]
pub extern "C" fn adpcm_leaky(input: *const u8, n_bytes: usize, out: *mut i16) -> usize {
    let mut pred: i32 = 0;
    let mut index: i32 = 0;
    let total = n_bytes * 2;
    let mut o: usize = 0;
    while o < total {
        let byte = unsafe { *input.add(o >> 1) } as i32;
        let nib = (byte >> (((o & 1) as i32) * 4)) & 15;
        let idx_delta = unsafe { *INDEX_TABLE.as_ptr().add((nib & 7) as usize) };
        index = (((index * 15) >> 4) + idx_delta + 2).clamp(0, 88);
        let step = unsafe { *STEP_TABLE.as_ptr().add(index as usize) };
        let mut diff = step >> 3;
        diff += ((nib & 4) >> 2) * step;
        diff += ((nib & 2) >> 1) * (step >> 1);
        diff += (nib & 1) * (step >> 2);
        let signed = 0i32.wrapping_sub(diff) & (0i32.wrapping_sub((nib >> 3) & 1));
        pred = (((pred * 15) >> 4) + diff + signed * 2).clamp(-32768, 32767);
        unsafe { *out.add(o) = pred as i16 };
        o += 1;
    }
    total
}

/// Кавычковый лексер построчного формата: state 0=вне, 1=в кавычках,
/// 2=escape. '\n' — жёсткий ресет в 0 из любого состояния: любые две
/// траектории сливаются на первом переводе строки (как у Mytkowicz).
#[no_mangle]
pub extern "C" fn fsm_lex(input: *const u8, n_bytes: usize, out: *mut u8) -> usize {
    let mut st: i32 = 0;
    let mut i: usize = 0;
    while i < n_bytes {
        let c = unsafe { *input.add(i) } as i32;
        let isq = c == 34;
        let isb = c == 92;
        let isn = c == 10;
        // st' = 0 при \n; 2 если st==1 и \; 0 если st==1 и "; 1 внутри; '"' снаружи -> 1
        let inner = (st == 1) & !isq & !isb;
        let next = (isn as i32) * 0
            + (((!isn) & (st == 1) & isb) as i32) * 2
            + (((!isn) & inner) as i32) * 1
            + (((!isn) & (st == 1) & isq) as i32) * 0
            + (((!isn) & (st == 2)) as i32) * 1
            + (((!isn) & (st == 0) & isq) as i32) * 1;
        st = next;
        unsafe { *out.add(i) = st as u8 };
        i += 1;
    }
    n_bytes
}

/// Сатурирующий аккумулятор: carried-сумма с клампом [-M, M].
/// Ошибка лечится на клампах — две траектории сливаются, когда обе
/// упираются в потолок. С biased-входом сливается быстро.
#[no_mangle]
pub extern "C" fn sat_acc(input: *const i16, n: usize, out: *mut i32) -> usize {
    let mut acc: i32 = 0;
    let mut i: usize = 0;
    while i < n {
        let x = unsafe { *input.add(i) } as i32;
        acc = (acc + x).clamp(-1_000_000, 1_000_000);
        unsafe { *out.add(i) = acc };
        i += 1;
    }
    n
}

/// Ниббл-автомат с match-диспетчеризацией (16 плотных рукавов -> br_table)
/// и переменным модулем (-> i32.rem_u). Carried: st + acc.
/// Класс символа определяет переход; синхронизирующий класс 15 -> st=0
/// из любого состояния.
#[no_mangle]
pub extern "C" fn nibble_fsm(input: *const u8, n_bytes: usize, out: *mut u8,
                             modulus: u32) -> usize {
    let mut st: u32 = 0;
    let mut acc: u32 = 0;
    let m = if modulus == 0 { 1 } else { modulus };
    let mut i: usize = 0;
    while i < n_bytes {
        let b = unsafe { *input.add(i) } as u32;
        let nib = b & 15;
        // match на 16 плотных значений -> rustc сгенерирует br_table
        let cls: u32 = match nib {
            0 => 15, 1 => 0, 2 => 1, 3 => 0, 4 => 2, 5 => 0, 6 => 3,
            7 => 0, 8 => 4, 9 => 0, 10 => 5, 11 => 0, 12 => 6, 13 => 0,
            14 => 7, 15 => 15,
            _ => 0,
        };
        if cls == 15 { st = 0; acc = 0 } else { st = (st + cls) & 7 };
        acc = (acc * 33 + nib) % m;
        unsafe { *out.add(i) = ((st << 4) | (acc & 15)) as u8 };
        i += 1;
    }
    n_bytes
}

/// Мультипликативная хеш-цепочка: h carried, ошибка не затухает никогда.
/// Честная стена — контрольный эксперимент.
#[no_mangle]
pub extern "C" fn hash_chain(input: *const u8, n_bytes: usize, out: *mut u8) -> usize {
    let mut h: u32 = 2166136261;
    let mut i: usize = 0;
    while i < n_bytes {
        let c = unsafe { *input.add(i) } as u32;
        h ^= c;
        h = h.wrapping_mul(16777619);
        h ^= h >> 13;
        unsafe { *out.add(i) = (h >> 24) as u8 };
        i += 1;
    }
    n_bytes
}

/// f32 IIR-фильтр: y = 0.875*y + x — контрактивная динамика на f32
/// (lambda = 7/8). Покрывает f32.mul/add/convert/trunc_sat/store.
#[no_mangle]
pub extern "C" fn float_leak(input: *const u8, n_bytes: usize, out: *mut u8) -> usize {
    let mut y: f32 = 0.0;
    let mut i: usize = 0;
    while i < n_bytes {
        let x = unsafe { *input.add(i) } as f32;
        y = y * 0.875 + x;
        let v = if y > 255.0 { 255.0 } else if y < 0.0 { 0.0 } else { y };
        unsafe { *out.add(i) = v as u8 };
        i += 1;
    }
    n_bytes
}

/// Проверка инлайнинга call: хеллер без inline (rustc эмитит настоящий
/// call 0x10). Контрактивная динамика: st = (st + class(x)) & 7 с reset.
#[inline(never)]
fn nib_class(v: u8) -> u32 {
    ((v.wrapping_mul(73) >> 5) & 7) as u32
}

#[no_mangle]
pub extern "C" fn call_test(input: *const u8, n_bytes: usize, out: *mut u8) -> usize {
    let mut st: u32 = 0;
    let mut i: usize = 0;
    while i < n_bytes {
        let c = unsafe { *input.add(i) };
        let cls = nib_class(c);
        st = if cls == 7 { 0 } else { (st + cls) & 7 };
        unsafe { *out.add(i) = ((st << 4) | cls) as u8 };
        i += 1;
    }
    n_bytes
}

/// Реальная DSP-цепочка (аналог `ffmpeg -af highpass,acompressor`):
/// HP80 -> HP30 -> peak-hold envelope (VU) -> gain -> out. Четыре
/// carried f32-аккумулятора; все полюса |lambda| < 1 => сжатие.
#[no_mangle]
pub extern "C" fn audio_chain(input: *const u8, n_bytes: usize, out: *mut u8) -> usize {
    let n = n_bytes / 4;
    let (mut y1, mut x1) = (0f32, 0f32);
    let (mut y2, mut x2) = (0f32, 0f32);
    let mut env = 0f32;
    let mut i: usize = 0;
    while i < n {
        let x = f32::from_bits(unsafe { (input as *const u32).add(i).read() });
        let a = x - x1 + 0.9886 * y1; x1 = x; y1 = a;
        let b = a - x2 + 0.9957 * y2; x2 = a; y2 = b;
        let m = if b < 0.0 { -b } else { b };
        env = if m > env { m } else { env * 0.995 };
        let g = if env > 0.5 { 0.5 / env } else { 1.0 };
        let o = b * g;
        unsafe { (out as *mut u32).add(i).write(o.to_bits()) };
        i += 1;
    }
    n * 4
}
