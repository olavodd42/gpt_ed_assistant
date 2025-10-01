def linearize(row):
    return (
        f"Age: {row.age} | Sex: {row.sex} | HR: {row.hr} | RR: {row.rr} | "
        f"O2Sat: {row.o2sat} | Chief: {row.chief_complaint} | "
        f"CBC: WBC={row.wbc}, Hb={row.hb} | "
        f"CHEM: Na={row.na}, K={row.k}, Creatinine={row.creatinine}"
    )
