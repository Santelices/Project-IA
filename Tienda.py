import ShoppingIA as shop

# Shop
def main():
    class_shop = shop.ShopIA()
    cap = class_shop.init()
    # Stream
    stream = class_shop.tiendaIA(cap)

if __name__ == "__main__":
    main()


