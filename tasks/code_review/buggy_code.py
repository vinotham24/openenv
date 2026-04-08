def calculate_invoice_total(items, tax_rate=0.1, discount=0.0):
    subtotal = 0
    for item in items:
        subtotal += item["price"] * item["quantity"]

    if discount:
        subtotal = subtotal - discount * 100

    taxed_total = subtotal + tax_rate
    return round(taxed_total, 2)


def summarize_orders(orders):
    summary = {}
    for order in orders:
        customer = order["customer"]
        summary[customer] = summary.get(customer, 0) + 1
    return sorted(summary.items(), key=lambda pair: pair[0], reverse=True)
