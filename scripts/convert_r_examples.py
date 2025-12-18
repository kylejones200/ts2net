import argparse, sys
from ts2net_py.rcompat import r_to_panel_csv, list_r_objects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--object", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--name-col", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k in list_r_objects(a.input):
            print(k)
        return 0
    print(r_to_panel_csv(a.input, a.object, a.output, name_col=a.name_col))
    return 0


if __name__ == "__main__":
    sys.exit(main())
