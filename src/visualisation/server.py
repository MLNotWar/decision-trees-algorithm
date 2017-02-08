from flask import Flask, render_template, jsonify


TEST_DATA = {
    "name": "root",
    "rule": "null",
    "children": [{
        "name": "5",
        "rule": "test"
    }, {
        "name": "2",
        "rule": "sunny",
        "children": [{
            "name": "no",
            "rule": "high",
        }, {
            "name": "yes",
            "rule": "normal",
        }]
    }, {
        "name": "yes",
        "rule": "overcast",
    }, {
        "name": "3",
        "rule": "rainy",
        "children": [{
            "name": "no",
            "rule": "TRUE",
        }, {
            "name": "yes",
            "rule": "FALSE",
        }]
    }]
}


def tag(data):
    def walk(_node):
        yield _node

        if "children" in _node:
            for _child in _node["children"]:
                yield from walk(_child)

    n = 0
    for node in walk(data):
        if "id" not in node:
            node["id"] = n
            n += 1

    return data


def main(data, **options):
    data = tag(data)
    app = Flask("visualisation")

    @app.route("/")
    def index():
        return render_template("app.html")

    @app.route("/data")
    def _():
        return jsonify(data)

    app.run(**options)


if __name__ == "__main__":
    main(TEST_DATA)
