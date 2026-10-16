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
